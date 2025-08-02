package provider

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/charmbracelet/catwalk/pkg/catwalk"
	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/llm/tools"
	"github.com/charmbracelet/crush/internal/log"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

type openaiClient struct {
	providerOptions providerClientOptions
	client          openai.Client
}

type OpenAIClient ProviderClient

func newOpenAIClient(opts providerClientOptions) OpenAIClient {
	return &openaiClient{
		providerOptions: opts,
		client:          createOpenAIClient(opts),
	}
}

func createOpenAIClient(opts providerClientOptions) openai.Client {
	openaiClientOptions := []option.RequestOption{}
	if opts.apiKey != "" {
		openaiClientOptions = append(openaiClientOptions, option.WithAPIKey(opts.apiKey))
	}
	if opts.baseURL != "" {
		resolvedBaseURL, err := config.Get().Resolve(opts.baseURL)
		if err == nil {
			openaiClientOptions = append(openaiClientOptions, option.WithBaseURL(resolvedBaseURL))
		}
	}
	if config.Get().Options.Debug {
		httpClient := log.NewHTTPClient()
		openaiClientOptions = append(openaiClientOptions, option.WithHTTPClient(httpClient))
	}
	for k, v := range opts.extraHeaders {
		openaiClientOptions = append(openaiClientOptions, option.WithHeader(k, v))
	}
	for k, v := range opts.extraBody {
		openaiClientOptions = append(openaiClientOptions, option.WithJSONSet(k, v))
	}
	return openai.NewClient(openaiClientOptions...)
}

func (o *openaiClient) convertMessages(messages []message.Message) []responses.ResponseInputItemUnionParam {
	items := make([]responses.ResponseInputItemUnionParam, 0)
	for _, msg := range messages {
		switch msg.Role {
		case message.User, message.Assistant:
			contentList := responses.ResponseInputMessageContentListParam{}
			text := msg.Content().String()
			if text != "" {
				contentList = append(contentList, responses.ResponseInputContentParamOfInputText(text))
			}
			for _, binary := range msg.BinaryContent() {
				img := responses.ResponseInputImageParam{Detail: responses.ResponseInputImageDetailAuto}
				img.ImageURL = param.NewOpt(binary.String(catwalk.InferenceProviderOpenAI))
				contentList = append(contentList, responses.ResponseInputContentUnionParam{OfInputImage: &img})
			}
			if len(contentList) > 0 {
				role := responses.EasyInputMessageRole(string(msg.Role))
				items = append(items, responses.ResponseInputItemParamOfMessage(contentList, role))
			}
			if msg.Role == message.Assistant {
				for _, call := range msg.ToolCalls() {
					items = append(items, responses.ResponseInputItemParamOfFunctionCall(call.Input, call.ID, call.Name))
				}
			}
		case message.Tool:
			for _, result := range msg.ToolResults() {
				items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(result.ToolCallID, result.Content))
			}
		}
	}
	return items
}

func (o *openaiClient) convertTools(toolsList []tools.BaseTool) []responses.ToolUnionParam {
	openaiTools := make([]responses.ToolUnionParam, len(toolsList))
	for i, t := range toolsList {
		info := t.Info()
		fn := responses.FunctionToolParam{
			Name: info.Name,
			Parameters: map[string]any{
				"type":       "object",
				"properties": info.Parameters,
				"required":   info.Required,
			},
			Strict: param.NewOpt(true),
		}
		if info.Description != "" {
			fn.Description = param.NewOpt(info.Description)
		}
		openaiTools[i] = responses.ToolUnionParam{OfFunction: &fn}
	}
	return openaiTools
}

func (o *openaiClient) preparedParams(input []responses.ResponseInputItemUnionParam, tools []responses.ToolUnionParam) responses.ResponseNewParams {
	model := o.providerOptions.model(o.providerOptions.modelType)
	cfg := config.Get()
	modelConfig := cfg.Models[config.SelectedModelTypeLarge]
	if o.providerOptions.modelType == config.SelectedModelTypeSmall {
		modelConfig = cfg.Models[config.SelectedModelTypeSmall]
	}
	maxTokens := model.DefaultMaxTokens
	if modelConfig.MaxTokens > 0 {
		maxTokens = modelConfig.MaxTokens
	}
	if o.providerOptions.maxTokens > 0 {
		maxTokens = o.providerOptions.maxTokens
	}
	params := responses.ResponseNewParams{
		Model:           responses.ResponsesModel(model.ID),
		Input:           responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Tools:           tools,
		Instructions:    param.NewOpt(o.providerOptions.systemMessage),
		MaxOutputTokens: param.NewOpt(maxTokens),
	}
	if o.providerOptions.disableCache {
		params.Store = param.NewOpt(false)
	}
	if model.CanReason {
		params.Reasoning = shared.ReasoningParam{Effort: shared.ReasoningEffort(modelConfig.ReasoningEffort)}
	}
	return params
}

func (o *openaiClient) send(ctx context.Context, messages []message.Message, tools []tools.BaseTool) (*ProviderResponse, error) {
	params := o.preparedParams(o.convertMessages(messages), o.convertTools(tools))
	attempts := 0
	for {
		attempts++
		resp, err := o.client.Responses.New(ctx, params)
		if err != nil {
			retry, after, retryErr := o.shouldRetry(attempts, err)
			if retryErr != nil {
				return nil, retryErr
			}
			if retry {
				slog.Warn("Retrying due to rate limit", "attempt", attempts, "max_retries", maxRetries)
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(time.Duration(after) * time.Millisecond):
					continue
				}
			}
			return nil, retryErr
		}
		content := resp.OutputText()
		toolCalls := o.toolCalls(*resp)
		finish := message.FinishReasonEndTurn
		if len(toolCalls) > 0 {
			finish = message.FinishReasonToolUse
		}
		return &ProviderResponse{
			Content:      content,
			ToolCalls:    toolCalls,
			Usage:        o.usage(*resp),
			FinishReason: finish,
		}, nil
	}
}

func (o *openaiClient) stream(ctx context.Context, messages []message.Message, tools []tools.BaseTool) <-chan ProviderEvent {
	params := o.preparedParams(o.convertMessages(messages), o.convertTools(tools))
	attempts := 0
	eventChan := make(chan ProviderEvent)

	go func() {
		for {
			attempts++
			stream := o.client.Responses.NewStreaming(ctx, params)
			currentContent := ""
			toolCalls := []message.ToolCall{}
			var currentTool *message.ToolCall

			for stream.Next() {
				ev := stream.Current()
				switch v := ev.AsAny().(type) {
				case responses.ResponseTextDeltaEvent:
					if v.Delta != "" {
						eventChan <- ProviderEvent{Type: EventContentDelta, Content: v.Delta}
						currentContent += v.Delta
					}
				case responses.ResponseFunctionCallArgumentsDeltaEvent:
					if currentTool == nil {
						currentTool = &message.ToolCall{ID: v.CallID, Name: v.Name, Finished: false}
						eventChan <- ProviderEvent{Type: EventToolUseStart, ToolCall: currentTool}
					}
					currentTool.Input += v.Arguments
					eventChan <- ProviderEvent{Type: EventToolUseDelta, ToolCall: &message.ToolCall{ID: currentTool.ID, Name: currentTool.Name, Input: v.Arguments}}
				case responses.ResponseFunctionCallArgumentsDoneEvent:
					if currentTool != nil {
						currentTool.Finished = true
						toolCalls = append(toolCalls, *currentTool)
						eventChan <- ProviderEvent{Type: EventToolUseStop, ToolCall: currentTool}
						currentTool = nil
					}
				case responses.ResponseCompletedEvent:
					finish := message.FinishReasonEndTurn
					if len(toolCalls) > 0 {
						finish = message.FinishReasonToolUse
					}
					eventChan <- ProviderEvent{
						Type: EventComplete,
						Response: &ProviderResponse{
							Content:      currentContent,
							ToolCalls:    toolCalls,
							Usage:        o.usage(v.Response),
							FinishReason: finish,
						},
					}
					close(eventChan)
					return
				case responses.ResponseErrorEvent:
					eventChan <- ProviderEvent{Type: EventError, Error: errors.New(v.Error.Message)}
					close(eventChan)
					return
				}
			}

			if err := stream.Err(); err != nil {
				retry, after, retryErr := o.shouldRetry(attempts, err)
				if retryErr != nil {
					eventChan <- ProviderEvent{Type: EventError, Error: retryErr}
					close(eventChan)
					return
				}
				if retry {
					slog.Warn("Retrying due to rate limit", "attempt", attempts, "max_retries", maxRetries)
					select {
					case <-ctx.Done():
						if ctx.Err() != nil {
							eventChan <- ProviderEvent{Type: EventError, Error: ctx.Err()}
						}
						close(eventChan)
						return
					case <-time.After(time.Duration(after) * time.Millisecond):
						continue
					}
				}
				eventChan <- ProviderEvent{Type: EventError, Error: retryErr}
				close(eventChan)
				return
			}
		}
	}()

	return eventChan
}

func (o *openaiClient) shouldRetry(attempts int, err error) (bool, int64, error) {
	if attempts > maxRetries {
		return false, 0, fmt.Errorf("maximum retry attempts reached for rate limit: %d retries", maxRetries)
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false, 0, err
	}
	var apiErr *openai.Error
	retryMs := 0
	retryAfterValues := []string{}
	if errors.As(err, &apiErr) {
		if apiErr.StatusCode == 401 {
			o.providerOptions.apiKey, err = config.Get().Resolve(o.providerOptions.config.APIKey)
			if err != nil {
				return false, 0, fmt.Errorf("failed to resolve API key: %w", err)
			}
			o.client = createOpenAIClient(o.providerOptions)
			return true, 0, nil
		}
		if apiErr.StatusCode != 429 && apiErr.StatusCode != 500 {
			return false, 0, err
		}
		retryAfterValues = apiErr.Response.Header.Values("Retry-After")
	}
	if apiErr != nil {
		slog.Warn("OpenAI API error", "status_code", apiErr.StatusCode, "message", apiErr.Message, "type", apiErr.Type)
		if len(retryAfterValues) > 0 {
			slog.Warn("Retry-After header", "values", retryAfterValues)
		}
	} else {
		slog.Error("OpenAI API error", "error", err.Error(), "attempt", attempts, "max_retries", maxRetries)
	}
	backoffMs := 2000 * (1 << (attempts - 1))
	jitterMs := int(float64(backoffMs) * 0.2)
	retryMs = backoffMs + jitterMs
	if len(retryAfterValues) > 0 {
		if _, err := fmt.Sscanf(retryAfterValues[0], "%d", &retryMs); err == nil {
			retryMs = retryMs * 1000
		}
	}
	return true, int64(retryMs), nil
}

func (o *openaiClient) toolCalls(resp responses.Response) []message.ToolCall {
	calls := []message.ToolCall{}
	for _, item := range resp.Output {
		fc := item.AsFunctionCall()
		if fc.CallID != "" {
			calls = append(calls, message.ToolCall{
				ID:       fc.CallID,
				Name:     fc.Name,
				Input:    fc.Arguments,
				Type:     "function",
				Finished: true,
			})
		}
	}
	return calls
}

func (o *openaiClient) usage(resp responses.Response) TokenUsage {
	cached := resp.Usage.InputTokensDetails.CachedTokens
	return TokenUsage{
		InputTokens:         resp.Usage.InputTokens - cached,
		OutputTokens:        resp.Usage.OutputTokens,
		CacheCreationTokens: 0,
		CacheReadTokens:     cached,
	}
}

func (o *openaiClient) Model() catwalk.Model {
	return o.providerOptions.model(o.providerOptions.modelType)
}
