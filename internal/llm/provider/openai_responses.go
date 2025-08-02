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
	"github.com/charmbracelet/crush/internal/message"
	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// openaiResponsesClient implements ProviderClient using the OpenAI Responses API.
type openaiResponsesClient struct {
	providerOptions providerClientOptions
	client          openai.Client
}

// OpenAIResponsesClient defines the interface for the responses provider.
type OpenAIResponsesClient ProviderClient

func newOpenAIResponsesClient(opts providerClientOptions) OpenAIResponsesClient {
	return &openaiResponsesClient{
		providerOptions: opts,
		client:          createOpenAIClient(opts),
	}
}

func (o *openaiResponsesClient) convertMessages(messages []message.Message) (items []responses.ResponseInputItemUnionParam) {
	systemMessage := o.providerOptions.systemMessage
	if o.providerOptions.systemPromptPrefix != "" {
		systemMessage = o.providerOptions.systemPromptPrefix + "\n" + systemMessage
	}
	items = append(items, responses.ResponseInputItemParamOfMessage(systemMessage, responses.EasyInputMessageRoleSystem))

	for _, msg := range messages {
		switch msg.Role {
		case message.User:
			if len(msg.BinaryContent()) == 0 && len(msg.ImageURLContent()) == 0 {
				items = append(items, responses.ResponseInputItemParamOfMessage(msg.Content().String(), responses.EasyInputMessageRoleUser))
				continue
			}
			contentList := responses.ResponseInputMessageContentListParam{}
			if text := msg.Content().String(); text != "" {
				contentList = append(contentList, responses.ResponseInputContentUnionParam{OfInputText: &responses.ResponseInputTextParam{Text: text}})
			}
			for _, b := range msg.BinaryContent() {
				img := responses.ResponseInputImageParam{
					ImageURL: param.NewOpt(b.String(catwalk.InferenceProviderOpenAI)),
					Detail:   responses.ResponseInputImageDetailAuto,
				}
				contentList = append(contentList, responses.ResponseInputContentUnionParam{OfInputImage: &img})
			}
			for _, imgURL := range msg.ImageURLContent() {
				img := responses.ResponseInputImageParam{
					ImageURL: param.NewOpt(imgURL.URL),
					Detail:   responses.ResponseInputImageDetailAuto,
				}
				if imgURL.Detail != "" {
					img.Detail = responses.ResponseInputImageDetail(imgURL.Detail)
				}
				contentList = append(contentList, responses.ResponseInputContentUnionParam{OfInputImage: &img})
			}
			items = append(items, responses.ResponseInputItemParamOfMessage(contentList, responses.EasyInputMessageRoleUser))
		case message.Assistant:
			if text := msg.Content().String(); text != "" {
				items = append(items, responses.ResponseInputItemParamOfMessage(text, responses.EasyInputMessageRoleAssistant))
			}
			for _, call := range msg.ToolCalls() {
				items = append(items, responses.ResponseInputItemParamOfFunctionCall(call.Input, call.ID, call.Name))
			}
		case message.Tool:
			for _, result := range msg.ToolResults() {
				items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(result.ToolCallID, result.Content))
			}
		}
	}
	return
}

func (o *openaiResponsesClient) convertTools(toolsList []tools.BaseTool) []responses.ToolUnionParam {
	toolsParams := make([]responses.ToolUnionParam, len(toolsList))
	for i, t := range toolsList {
		info := t.Info()
		fn := responses.FunctionToolParam{
			Name:        info.Name,
			Description: param.NewOpt(info.Description),
			Parameters: map[string]any{
				"type":       "object",
				"properties": info.Parameters,
				"required":   info.Required,
			},
			Strict: param.NewOpt(true),
		}
		toolsParams[i] = responses.ToolUnionParam{OfFunction: &fn}
	}
	return toolsParams
}

func (o *openaiResponsesClient) finishReason(r responses.Response) message.FinishReason {
	if r.IncompleteDetails.Reason != "" {
		switch r.IncompleteDetails.Reason {
		case "max_output_tokens":
			return message.FinishReasonMaxTokens
		default:
			return message.FinishReasonError
		}
	}
	if len(o.toolCalls(r)) > 0 {
		return message.FinishReasonToolUse
	}
	return message.FinishReasonEndTurn
}

func (o *openaiResponsesClient) preparedParams(messages []responses.ResponseInputItemUnionParam, tools []responses.ToolUnionParam) responses.ResponseNewParams {
	model := o.providerOptions.model(o.providerOptions.modelType)
	cfg := config.Get()
	modelConfig := cfg.Models[config.SelectedModelTypeLarge]
	if o.providerOptions.modelType == config.SelectedModelTypeSmall {
		modelConfig = cfg.Models[config.SelectedModelTypeSmall]
	}
	reasoningEffort := modelConfig.ReasoningEffort

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(model.ID),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: responses.ResponseInputParam(messages)},
		Tools: tools,
	}
	maxTokens := model.DefaultMaxTokens
	if modelConfig.MaxTokens > 0 {
		maxTokens = modelConfig.MaxTokens
	}
	if o.providerOptions.maxTokens > 0 {
		maxTokens = o.providerOptions.maxTokens
	}
	params.MaxOutputTokens = param.NewOpt(maxTokens)
	if model.CanReason {
		var effort shared.ReasoningEffort
		switch reasoningEffort {
		case "low":
			effort = shared.ReasoningEffortLow
		case "medium":
			effort = shared.ReasoningEffortMedium
		case "high":
			effort = shared.ReasoningEffortHigh
		default:
			effort = shared.ReasoningEffort(reasoningEffort)
		}
		params.Reasoning = shared.ReasoningParam{Effort: effort}
	}
	return params
}

func (o *openaiResponsesClient) usage(resp responses.Response) TokenUsage {
	usage := TokenUsage{
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
	}
	usage.CacheReadTokens = resp.Usage.InputTokensDetails.CachedTokens
	return usage
}

func (o *openaiResponsesClient) toolCalls(resp responses.Response) []message.ToolCall {
	var calls []message.ToolCall
	for _, item := range resp.Output {
		fc := item.AsFunctionCall()
		if fc.CallID != "" {
			calls = append(calls, message.ToolCall{
				ID:    fc.CallID,
				Name:  fc.Name,
				Input: fc.Arguments,
				Type:  "function",
			})
		}
	}
	return calls
}

func (o *openaiResponsesClient) send(ctx context.Context, messages []message.Message, tools []tools.BaseTool) (*ProviderResponse, error) {
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
		calls := o.toolCalls(*resp)
		finish := o.finishReason(*resp)
		if len(calls) > 0 {
			finish = message.FinishReasonToolUse
		}
		return &ProviderResponse{
			Content:      content,
			ToolCalls:    calls,
			Usage:        o.usage(*resp),
			FinishReason: finish,
		}, nil
	}
}

func (o *openaiResponsesClient) stream(ctx context.Context, messages []message.Message, tools []tools.BaseTool) <-chan ProviderEvent {
	eventChan := make(chan ProviderEvent, 2)
	go func() {
		defer close(eventChan)
		resp, err := o.send(ctx, messages, tools)
		if err != nil {
			eventChan <- ProviderEvent{Type: EventError, Error: err}
			return
		}
		if resp.Content != "" {
			eventChan <- ProviderEvent{Type: EventContentDelta, Content: resp.Content}
		}
		eventChan <- ProviderEvent{Type: EventComplete, Response: resp}
	}()
	return eventChan
}

func (o *openaiResponsesClient) shouldRetry(attempts int, err error) (bool, int64, error) {
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

func (o *openaiResponsesClient) Model() catwalk.Model {
	return o.providerOptions.model(o.providerOptions.modelType)
}
