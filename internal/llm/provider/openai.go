package provider

import (
	"context"
	"errors"
	"fmt"
	"io"
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
		if resolved, err := config.Get().Resolve(opts.baseURL); err == nil {
			openaiClientOptions = append(openaiClientOptions, option.WithBaseURL(resolved))
		}
	}
	if cfg := config.Get(); cfg != nil && cfg.Options.Debug {
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

func (o *openaiClient) convertMessages(messages []message.Message) responses.ResponseInputParam {
	var input responses.ResponseInputParam
	for _, msg := range messages {
		switch msg.Role {
		case message.User:
			var content responses.ResponseInputMessageContentListParam
			if text := msg.Content().String(); text != "" {
				content = append(content, responses.ResponseInputContentParamOfInputText(text))
			}
			for _, binary := range msg.BinaryContent() {
				img := responses.ResponseInputImageParam{Detail: responses.ResponseInputImageDetailAuto}
				img.ImageURL = param.NewOpt(binary.String(catwalk.InferenceProviderOpenAI))
				content = append(content, responses.ResponseInputContentUnionParam{OfInputImage: &img})
			}
			for _, url := range msg.ImageURLContent() {
				img := responses.ResponseInputImageParam{Detail: responses.ResponseInputImageDetail(url.Detail)}
				if img.Detail == "" {
					img.Detail = responses.ResponseInputImageDetailAuto
				}
				img.ImageURL = param.NewOpt(url.URL)
				content = append(content, responses.ResponseInputContentUnionParam{OfInputImage: &img})
			}
			input = append(input, responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRoleUser))

		case message.Assistant:
			if text := msg.Content().String(); text != "" {
				content := responses.ResponseInputMessageContentListParam{
					responses.ResponseInputContentParamOfInputText(text),
				}
				input = append(input, responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRoleAssistant))
			}
			for _, call := range msg.ToolCalls() {
				fc := responses.ResponseFunctionToolCallParam{
					CallID:    call.ID,
					Name:      call.Name,
					Arguments: call.Input,
				}
				input = append(input, responses.ResponseInputItemUnionParam{OfFunctionCall: &fc})
			}

		case message.Tool:
			for _, result := range msg.ToolResults() {
				out := responses.ResponseInputItemFunctionCallOutputParam{
					CallID: result.ToolCallID,
					Output: result.Content,
				}
				input = append(input, responses.ResponseInputItemUnionParam{OfFunctionCallOutput: &out})
			}
		}
	}
	return input
}

func (o *openaiClient) convertTools(tools []tools.BaseTool) []responses.ToolUnionParam {
	openaiTools := make([]responses.ToolUnionParam, len(tools))
	for i, tool := range tools {
		info := tool.Info()
		function := responses.ToolParamOfFunction(info.Name, map[string]any{
			"type":       "object",
			"properties": info.Parameters,
			"required":   info.Required,
		}, true)
		if function.OfFunction != nil {
			function.OfFunction.Description = param.NewOpt(info.Description)
		}
		openaiTools[i] = function
	}
	return openaiTools
}

func (o *openaiClient) finishReason(status responses.ResponseStatus) message.FinishReason {
	switch status {
	case responses.ResponseStatusCompleted:
		return message.FinishReasonEndTurn
	case responses.ResponseStatusIncomplete:
		return message.FinishReasonMaxTokens
	case responses.ResponseStatusCancelled:
		return message.FinishReasonCanceled
	case responses.ResponseStatusFailed:
		return message.FinishReasonError
	default:
		return message.FinishReasonUnknown
	}
}

func (o *openaiClient) preparedParams(messages responses.ResponseInputParam, tools []responses.ToolUnionParam) responses.ResponseNewParams {
	model := o.providerOptions.model(o.providerOptions.modelType)
	cfg := config.Get()

	var modelConfig config.SelectedModel
	if cfg != nil {
		modelConfig = cfg.Models[config.SelectedModelTypeLarge]
		if o.providerOptions.modelType == config.SelectedModelTypeSmall {
			modelConfig = cfg.Models[config.SelectedModelTypeSmall]
		}
	}

	systemMessage := o.providerOptions.systemMessage
	if o.providerOptions.systemPromptPrefix != "" {
		systemMessage = o.providerOptions.systemPromptPrefix + "\n" + systemMessage
	}

	params := responses.ResponseNewParams{
		Model:        shared.ResponsesModel(model.ID),
		Instructions: param.NewOpt(systemMessage),
		Input:        responses.ResponseNewParamsInputUnion{OfInputItemList: messages},
		Tools:        tools,
	}

	maxTokens := model.DefaultMaxTokens
	if modelConfig.MaxTokens > 0 {
		maxTokens = modelConfig.MaxTokens
	}
	if o.providerOptions.maxTokens > 0 {
		maxTokens = o.providerOptions.maxTokens
	}
	params.MaxOutputTokens = param.NewOpt(int64(maxTokens))

	if model.CanReason {
		effort := shared.ReasoningEffort(modelConfig.ReasoningEffort)
		params.Reasoning = shared.ReasoningParam{Effort: effort}
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

		content := ""
		for _, item := range resp.Output {
			if item.Type == "message" {
				msg := item.AsMessage()
				for _, part := range msg.Content {
					if part.Type == "output_text" {
						text := part.AsOutputText()
						content += text.Text
					}
				}
			}
		}
		toolCalls := o.toolCalls(*resp)
		finish := o.finishReason(resp.Status)
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
	eventChan := make(chan ProviderEvent)

	go func() {
		attempts := 0
		for {
			attempts++
			stream := o.client.Responses.NewStreaming(ctx, params)
			currentContent := ""
			for stream.Next() {
				evt := stream.Current()
				switch evt.Type {
				case "response.output_text.delta":
					delta := evt.AsResponseOutputTextDelta()
					currentContent += delta.Delta
					eventChan <- ProviderEvent{Type: EventContentDelta, Content: delta.Delta}
				case "response.output_item.added":
					added := evt.AsResponseOutputItemAdded()
					if added.Item.Type == "function_call" {
						fc := added.Item.AsFunctionCall()
						eventChan <- ProviderEvent{
							Type: EventToolUseStart,
							ToolCall: &message.ToolCall{
								ID:       fc.CallID,
								Name:     fc.Name,
								Finished: false,
							},
						}
					}
				case "response.completed":
					completed := evt.AsResponseCompleted()
					toolCalls := o.toolCalls(completed.Response)
					finish := o.finishReason(completed.Response.Status)
					if len(toolCalls) > 0 {
						finish = message.FinishReasonToolUse
					}
					eventChan <- ProviderEvent{
						Type: EventComplete,
						Response: &ProviderResponse{
							Content:      currentContent,
							ToolCalls:    toolCalls,
							Usage:        o.usage(completed.Response),
							FinishReason: finish,
						},
					}
				}
			}
			err := stream.Err()
			if err == nil || errors.Is(err, io.EOF) {
				break
			}
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
		close(eventChan)
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
			var e error
			o.providerOptions.apiKey, e = config.Get().Resolve(o.providerOptions.config.APIKey)
			if e != nil {
				return false, 0, fmt.Errorf("failed to resolve API key: %w", e)
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
		if _, e := fmt.Sscanf(retryAfterValues[0], "%d", &retryMs); e == nil {
			retryMs = retryMs * 1000
		}
	}
	return true, int64(retryMs), nil
}

func (o *openaiClient) toolCalls(resp responses.Response) []message.ToolCall {
	var calls []message.ToolCall
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			fc := item.AsFunctionCall()
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
	input := resp.Usage.InputTokens - cached
	return TokenUsage{
		InputTokens:         input,
		OutputTokens:        resp.Usage.OutputTokens,
		CacheCreationTokens: 0,
		CacheReadTokens:     cached,
	}
}

func (o *openaiClient) Model() catwalk.Model {
	return o.providerOptions.model(o.providerOptions.modelType)
}
