package llm

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/ollama/ollama/api"
)

// OllamaLLM implements the LLM interface for Ollama
type OllamaLLM struct {
	client *api.Client
}

// NewOllamaLLM creates a new Ollama LLM client
func NewOllamaLLM(host string) (*OllamaLLM, error) {
	baseURL, err := url.Parse(host)
	if err != nil {
		return nil, fmt.Errorf("invalid Ollama host URL: %w", err)
	}

	client := api.NewClient(baseURL, nil) // Using default http.Client
	return &OllamaLLM{client: client}, nil
}

// convertToOllamaMessages converts our generic Message type to Ollama's message type
func convertToOllamaMessages(messages []InputMessage) []api.Message {
	ollamaMessages := make([]api.Message, 0, len(messages))
	for _, msg := range messages {
		role := strings.ToLower(string(msg.Role))
		content := ""
		
		// Combine text content from MultiContent
		var textParts []string
		var images []api.ImageData
		for _, part := range msg.MultiContent {
			switch part.Type {
			case ContentTypeText:
				textParts = append(textParts, part.Text)
			case ContentTypeImage:
				if imageBytes, err := base64.StdEncoding.DecodeString(part.Data); err == nil {
					images = append(images, imageBytes)
				}
			}
		}
		content = strings.Join(textParts, "")

		ollamaMessages = append(ollamaMessages, api.Message{
			Role:    role,
			Content: content,
			Images:  images,
		})
	}
	return ollamaMessages
}

// convertFromOllamaMessage converts Ollama's message to our OutputMessage type
func convertFromOllamaMessage(msg api.Message) OutputMessage {
	return OutputMessage{
		Role:    Role(msg.Role),
		Content: msg.Content,
	}
}

// CreateChatCompletion implements the LLM interface for Ollama
func (o *OllamaLLM) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	ollamaReq := &api.ChatRequest{
		Model:    string(req.Model),
		Messages: convertToOllamaMessages(req.Messages),
		Options: map[string]interface{}{
			"temperature": req.Temperature,
		},
	}

	if req.TopP != nil {
		ollamaReq.Options["top_p"] = *req.TopP
	}

	if req.SystemPrompt != nil {
		ollamaReq.Options["system"] = *req.SystemPrompt
	}

	var finalResponse api.ChatResponse
	err := o.client.Chat(ctx, ollamaReq, func(response api.ChatResponse) error {
		finalResponse = response
		return nil
	})
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("Ollama chat completion failed: %w", err)
	}

	return ChatCompletionResponse{
		Choices: []Choice{
			{
				Index:   0,
				Message: convertFromOllamaMessage(finalResponse.Message),
				FinishReason: func() FinishReason {
					switch finalResponse.DoneReason {
					case "stop":
						return FinishReasonStop
					case "length":
						return FinishReasonMaxTokens
					default:
						return FinishReasonStop
					}
				}(),
			},
		},
		Usage: Usage{
			PromptTokens:     finalResponse.PromptEvalCount,
			CompletionTokens: finalResponse.EvalCount,
			TotalTokens:      finalResponse.PromptEvalCount + finalResponse.EvalCount,
		},
	}, nil
}

// ollamaStreamWrapper wraps Ollama's streaming to implement ChatCompletionStream
type ollamaStreamWrapper struct {
	ctx     context.Context
	client  *api.Client
	request *api.ChatRequest
	done    bool
}

func (w *ollamaStreamWrapper) Recv() (ChatCompletionResponse, error) {
	if w.done {
		return ChatCompletionResponse{}, io.EOF
	}

	var response ChatCompletionResponse
	err := w.client.Chat(w.ctx, w.request, func(resp api.ChatResponse) error {
		response = ChatCompletionResponse{
			Choices: []Choice{
				{
					Index: 0,
					Message: OutputMessage{
						Role:    Role(resp.Message.Role),
						Content: resp.Message.Content,
					},
					FinishReason: func() FinishReason {
						if resp.Done {
							return FinishReasonStop
						}
						return FinishReasonNull
					}(),
				},
			},
			Usage: Usage{
				PromptTokens:     resp.PromptEvalCount,
				CompletionTokens: resp.EvalCount,
				TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
			},
		}

		if resp.Done {
			w.done = true
		}
		return nil
	})

	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("Ollama stream receive failed: %w", err)
	}

	return response, nil
}

func (w *ollamaStreamWrapper) Close() error {
	w.done = true
	return nil
}

// CreateChatCompletionStream implements streaming for Ollama
func (o *OllamaLLM) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (ChatCompletionStream, error) {
	ollamaReq := &api.ChatRequest{
		Model:    string(req.Model),
		Messages: convertToOllamaMessages(req.Messages),
		Options: map[string]interface{}{
			"temperature": req.Temperature,
		},
	}

	if req.TopP != nil {
		ollamaReq.Options["top_p"] = *req.TopP
	}

	if req.SystemPrompt != nil {
		ollamaReq.Options["system"] = *req.SystemPrompt
	}

	return &ollamaStreamWrapper{
		ctx:     ctx,
		client:  o.client,
		request: ollamaReq,
	}, nil
}
