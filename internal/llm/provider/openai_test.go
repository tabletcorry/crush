package provider

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/charmbracelet/catwalk/pkg/catwalk"
	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestOpenAIClientStream(t *testing.T) {
	// Create a mock server that returns a minimal Responses stream
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		deltaEvent := map[string]any{
			"type":            "response.output_text.delta",
			"content_index":   0,
			"delta":           "hi",
			"item_id":         "msg1",
			"logprobs":        []any{},
			"output_index":    0,
			"sequence_number": 0,
		}

		jsonData, _ := json.Marshal(deltaEvent)
		w.Write([]byte("data: " + string(jsonData) + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	// Create OpenAI client pointing to our mock server
	client := &openaiClient{
		providerOptions: providerClientOptions{
			modelType:     config.SelectedModelTypeLarge,
			apiKey:        "test-key",
			systemMessage: "test",
			model: func(config.SelectedModelType) catwalk.Model {
				return catwalk.Model{
					ID:   "test-model",
					Name: "test-model",
				}
			},
		},
		client: openai.NewClient(
			option.WithAPIKey("test-key"),
			option.WithBaseURL(server.URL),
		),
	}

	// Create test messages
	messages := []message.Message{
		{
			Role:  message.User,
			Parts: []message.ContentPart{message.TextContent{Text: "Hello"}},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	eventsChan := client.stream(ctx, messages, nil)

	// Collect events - this will panic without the bounds check
	for event := range eventsChan {
		t.Logf("Received event: %+v", event)
		if event.Type == EventError || event.Type == EventComplete {
			break
		}
	}
}
