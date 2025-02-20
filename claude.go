package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/liushuangls/go-anthropic/v2"
	"golang.org/x/oauth2/google"
)

// ClaudeLLM implements the LLM interface for Anthropic's Claude
type ClaudeLLM struct {
	client *anthropic.Client
}

type BetaVersion string

const (
	BetaTools2024_04_04          BetaVersion = "tools-2024-04-04"
	BetaTools2024_05_16          BetaVersion = "tools-2024-05-16"
	BetaPromptCaching2024_07_31  BetaVersion = "prompt-caching-2024-07-31"
	BetaMessageBatches2024_09_24 BetaVersion = "message-batches-2024-09-24"
	BetaTokenCounting2024_11_01  BetaVersion = "token-counting-2024-11-01"
	BetaMaxTokens35_2024_07_15   BetaVersion = "max-tokens-3-5-sonnet-2024-07-15"
)

type ClientOption func(*ClaudeLLM)

// NewAnthropicLLM creates a new Claude LLM client (via Anthropic API)
func NewAnthropicLLM(apiKey string) *ClaudeLLM {

	// activate all beta versions by default
	opts := []BetaVersion{BetaTools2024_04_04, BetaTools2024_05_16, BetaPromptCaching2024_07_31, BetaMessageBatches2024_09_24, BetaTokenCounting2024_11_01, BetaMaxTokens35_2024_07_15}

	anthropicOpts := make([]anthropic.ClientOption, len(opts))
	for i, opt := range opts {
		anthropicOpts[i] = anthropic.WithBetaVersion(anthropic.BetaVersion(opt))
	}

	client := anthropic.NewClient(apiKey, anthropicOpts...)

	return &ClaudeLLM{client: client}

}

// NewVertexLLM creates a new Claude LLM client (via Vertex AI custom integration)
func NewVertexLLM(credBytes []byte, projectID string, location string) *ClaudeLLM {

	// activate all beta versions by default
	opts := []BetaVersion{BetaTools2024_04_04, BetaTools2024_05_16, BetaPromptCaching2024_07_31, BetaMessageBatches2024_09_24, BetaTokenCounting2024_11_01, BetaMaxTokens35_2024_07_15}

	ts, err := google.JWTAccessTokenSourceWithScope(
		credBytes,
		"https://www.googleapis.com/auth/cloud-platform",
		"https://www.googleapis.com/auth/cloud-platform.read-only",
	)
	if err != nil {
		fmt.Println("Error creating token source:", err)
		return nil
	}

	token, err := ts.Token()
	if err != nil {
		fmt.Println("Error getting token:", err)
		return nil
	}

	fmt.Println("Using Vertex AI with token prefix:", token.AccessToken[:10], "...")

	// activate all beta versions by default
	opts = append(opts, BetaTools2024_04_04, BetaTools2024_05_16, BetaPromptCaching2024_07_31, BetaMessageBatches2024_09_24, BetaTokenCounting2024_11_01, BetaMaxTokens35_2024_07_15)

	betaOpts := make([]anthropic.ClientOption, len(opts))
	for i, opt := range opts {
		betaOpts[i] = anthropic.WithBetaVersion(anthropic.BetaVersion(opt))
	}

	anthropicOpts := append(betaOpts, anthropic.WithVertexAI(projectID, location))

	client := anthropic.NewClient(token.AccessToken, anthropicOpts...)
	return &ClaudeLLM{client: client}
}

// convertToClaudeMessages converts our generic InputMessage type to Anthropic's messages
func convertToClaudeMessages(messages []InputMessage) []anthropic.Message {
	claudeMessages := make([]anthropic.Message, 0, len(messages))
	for _, msg := range messages {
		var role anthropic.ChatRole
		switch msg.Role {
		case RoleUser:
			role = anthropic.RoleUser
		case RoleAssistant:
			role = anthropic.RoleAssistant
		case RoleTool:
			// For Anthropic, pass tool messages as role=User with a tool_result content block
			role = anthropic.RoleUser
		default:
			// skip unknown roles
			continue
		}

		claudeMessage := anthropic.Message{
			Role: anthropic.ChatRole(role),
		}

		claudeMessage.Content = convertToClaudeMessageContent(msg.MultiContent)

		if msg.Role == RoleTool && len(msg.ToolResults) > 0 {
			toolResult := convertToClaudeMessageContentToolResult(msg.ToolResults[0])
			claudeMessage.Content = []anthropic.MessageContent{
				{
					Type:                     anthropic.MessagesContentTypeToolResult,
					MessageContentToolResult: &toolResult,
				},
			}
		}

		// If it's an assistant message calling a tool
		if msg.Role == RoleAssistant && len(msg.ToolCalls) > 0 {
			toolCall := msg.ToolCalls[0]
			claudeMessage.Content = []anthropic.MessageContent{
				{
					Type: anthropic.MessagesContentTypeToolUse,
					MessageContentToolUse: anthropic.NewMessageContentToolUse(
						toolCall.ID,
						toolCall.Function.Name,
						json.RawMessage(toolCall.Function.Arguments),
					),
				},
			}
		}

		claudeMessages = append(claudeMessages, claudeMessage)
	}
	return claudeMessages
}

// convertToClaudeMessageContent transforms our list of ContentPart into anthropic.MessageContent slices
func convertToClaudeMessageContent(content []ContentPart) []anthropic.MessageContent {
	multiContent := make([]anthropic.MessageContent, 0, len(content))
	for _, part := range content {
		switch part.Type {
		case ContentTypeText:
			multiContent = append(multiContent, anthropic.NewTextMessageContent(part.Text))
		case ContentTypeImage:
			multiContent = append(multiContent, anthropic.NewImageMessageContent(
				anthropic.NewMessageContentSource(
					anthropic.MessagesContentSourceTypeBase64,
					part.MediaType,
					part.Data,
				),
			))
		}
	}
	return multiContent
}

func convertToClaudeMessageContentToolResult(toolResult ToolResult) anthropic.MessageContentToolResult {
	return anthropic.MessageContentToolResult{
		ToolUseID: &toolResult.ToolCallID,
		Content:   []anthropic.MessageContent{anthropic.NewTextMessageContent(toolResult.Result)},
		IsError:   &toolResult.IsError,
	}
}

// convertFromClaudeMessage converts an anthropic.MessagesResponse to our OutputMessage
func convertFromClaudeMessage(msg anthropic.MessagesResponse) OutputMessage {
	var content string
	var toolCalls []anthropic.MessageContentToolUse

	// Gather text parts and possible tool calls
	if len(msg.Content) > 0 {
		var textParts []string
		for _, part := range msg.Content {
			switch part.Type {
			case anthropic.MessagesContentTypeText:
				if part.Text != nil {
					textParts = append(textParts, *part.Text)
				}
			case anthropic.MessagesContentTypeToolUse:
				if part.MessageContentToolUse != nil {
					toolCalls = append(toolCalls, *part.MessageContentToolUse)
				}
			}
		}
		content = strings.Join(textParts, "")
	}

	return OutputMessage{
		Role:      Role(msg.Role),
		Content:   content,
		ToolCalls: convertFromClaudeToolCalls(toolCalls),
	}
}

// convertFromClaudeToolCalls converts anthropic tool calls to our ToolCall
func convertFromClaudeToolCalls(toolCalls []anthropic.MessageContentToolUse) []ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}

	calls := make([]ToolCall, len(toolCalls))
	for i, call := range toolCalls {
		calls[i] = ToolCall{
			ID:   call.ID,
			Type: "function",
			Function: ToolCallFunction{
				Name:      call.Name,
				Arguments: string(call.Input),
			},
		}
	}
	return calls
}

// convertToClaudeTools translates our Tool definitions into anthropic.ToolDefinition
func convertToClaudeTools(tools []Tool) []anthropic.ToolDefinition {
	if len(tools) == 0 {
		return nil
	}

	claudeTools := make([]anthropic.ToolDefinition, len(tools))
	for i, tool := range tools {
		claudeTools[i] = anthropic.ToolDefinition{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: tool.Function.Parameters,
		}
	}
	return claudeTools
}

// CreateChatCompletion implements the non-streaming LLM interface for Claude
func (c *ClaudeLLM) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	if !c.isSupported(req.Model) {
		return ChatCompletionResponse{}, fmt.Errorf("model %s is not available", req.Model)
	}
	model := anthropic.Model(req.Model)

	tools := convertToClaudeTools(req.Tools)

	var toolChoice *anthropic.ToolChoice

	if len(tools) > 0 {
		toolChoice = &anthropic.ToolChoice{Type: "auto"}
	}

	topP := float32(1)
	if req.TopP != nil {
		topP = *req.TopP
	}

	var systemPrompt string
	if req.SystemPrompt != nil {
		systemPrompt = *req.SystemPrompt
	}

	claudeReq := anthropic.MessagesRequest{
		Model:       model,
		Messages:    convertToClaudeMessages(req.Messages),
		System:      systemPrompt,
		Temperature: &req.Temperature,
		TopP:        &topP,
		Tools:       tools,
		Stream:      false,
		MaxTokens:   req.MaxTokens,
		ToolChoice:  toolChoice,
	}

	resp, err := c.client.CreateMessages(ctx, claudeReq)
	if err != nil {
		return ChatCompletionResponse{}, err
	}

	choices := make([]Choice, 1)
	msg := convertFromClaudeMessage(resp)
	choices[0] = Choice{
		Index:        0,
		Message:      msg,
		FinishReason: convertFromClaudeFinishReason(resp.StopReason),
	}

	return ChatCompletionResponse{
		ID:      resp.ID,
		Choices: choices,
		Usage: Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}, nil
}

func convertFromClaudeFinishReason(reason anthropic.MessagesStopReason) FinishReason {
	switch reason {
	case anthropic.MessagesStopReasonEndTurn:
		return FinishReasonStop
	case anthropic.MessagesStopReasonToolUse:
		return FinishReasonToolCalls
	case anthropic.MessagesStopReasonMaxTokens:
		return FinishReasonMaxTokens
	case anthropic.MessagesStopReasonStopSequence:
		return FinishReasonStop
	}
	return FinishReason(reason)
}

// isSupported checks if the model is recognized as a Claude-friendly model
func (c *ClaudeLLM) isSupported(model Model) bool {
	switch model {
	case ModelClaude2Dot0,
		ModelClaude2Dot1,
		ModelClaude3Opus20240229,
		ModelClaude3Sonnet20240229,
		ModelClaude3Dot5Sonnet20240620,
		ModelClaude3Dot5Sonnet20241022,
		ModelClaude3Dot5SonnetLatest,
		ModelClaude3Haiku20240307,
		ModelClaude3Dot5HaikuLatest,
		ModelClaude3Dot5Haiku20241022:
		return true
	default:
		return false
	}
}

// claudeStreamWrapper wraps Anthropic's streaming approach to implement ChatCompletionStream
type claudeStreamWrapper struct {
	eventsChan chan ChatCompletionResponse
	errChan    chan error
	cancelFunc context.CancelFunc
	done       bool
	mu         sync.Mutex
}

// Recv returns the next available partial or final ChatCompletionResponse.
// If streaming is complete or an error occurs, returns an error (possibly io.EOF).
func (w *claudeStreamWrapper) Recv() (ChatCompletionResponse, error) {
	resp, ok := <-w.eventsChan
	if !ok {
		// channel closed; check if there was an error
		select {
		case err := <-w.errChan:
			if err != nil {
				return ChatCompletionResponse{}, err
			}
			return ChatCompletionResponse{}, io.EOF
		default:
			return ChatCompletionResponse{}, io.EOF
		}
	}
	return resp, nil
}

// Close stops the stream and cleans up any resources
func (w *claudeStreamWrapper) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.done {
		return nil
	}
	w.done = true
	// Cancel the context if possible
	if w.cancelFunc != nil {
		w.cancelFunc()
	}
	// Drain channels
	close(w.eventsChan)
	close(w.errChan)
	return nil
}

// CreateChatCompletionStream implements streaming for Claude with callbacks
func (c *ClaudeLLM) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (ChatCompletionStream, error) {
	if !c.isSupported(req.Model) {
		return nil, fmt.Errorf("model %s is not available", req.Model)
	}
	model := anthropic.Model(req.Model)

	// We'll create a child context to cancel if needed
	ctxStream, cancel := context.WithCancel(ctx)

	// We'll push partial updates to eventsChan, and push errors to errChan
	eventsChan := make(chan ChatCompletionResponse, 10)
	errChan := make(chan error, 1)

	// We'll track partial text and partial tool calls
	// We'll accumulate them as content comes in
	var partialTextBuilder strings.Builder
	var toolCalls []ToolCall
	// var stopReason anthropic.MessagesStopReason

	wrapper := &claudeStreamWrapper{
		eventsChan: eventsChan,
		errChan:    errChan,
		cancelFunc: cancel,
	}

	topP := float32(1)
	if req.TopP != nil {
		topP = *req.TopP
	}

	var systemPrompt string
	if req.SystemPrompt != nil {
		systemPrompt = *req.SystemPrompt
	}

	// Build request for streaming
	streamReq := anthropic.MessagesStreamRequest{
		MessagesRequest: anthropic.MessagesRequest{
			Model:       model,
			Messages:    convertToClaudeMessages(req.Messages),
			System:      systemPrompt,
			Temperature: &req.Temperature,
			TopP:        &topP,
			Tools:       convertToClaudeTools(req.Tools),
			Stream:      true,
			MaxTokens:   req.MaxTokens,
		},

		OnError: func(e anthropic.ErrorResponse) {
			select {
			case errChan <- e.Error:
			default:
			}
		},

		OnPing: func(data anthropic.MessagesEventPingData) {
			// Normally we ignore Ping events or log them
		},

		OnMessageStart: func(d anthropic.MessagesEventMessageStartData) {
			// This indicates a new "assistant" message is starting
			partialTextBuilder.Reset()
			toolCalls = nil
		},

		OnContentBlockStart: func(d anthropic.MessagesEventContentBlockStartData) {
			// We'll do nothing special here. We wait for deltas
		},

		OnContentBlockDelta: func(d anthropic.MessagesEventContentBlockDeltaData) {
			// We only handle partial text or partial JSON for tool calls
			if d.Delta.Type == anthropic.MessagesContentTypeTextDelta && d.Delta.Text != nil {
				partialTextBuilder.WriteString(*d.Delta.Text)
				// Send partial response
				eventsChan <- ChatCompletionResponse{
					Choices: []Choice{{
						Index: 0,
						Message: OutputMessage{
							Role:    RoleAssistant,
							Content: *d.Delta.Text,
							// tool calls remain the same
							ToolCalls: toolCalls,
						},
						FinishReason: FinishReasonNull,
					}},
				}
			} else if d.Delta.Type == anthropic.MessagesContentTypeInputJsonDelta && d.Delta.PartialJson != nil {
				// For a tool call, accumulate partial JSON
				// We'll finalize it at content_block_stop
				// If we wanted partial function calls, we'd do something here
			}
		},

		OnContentBlockStop: func(d anthropic.MessagesEventContentBlockStopData, block anthropic.MessageContent) {
			// If the content block is a tool call, finalize its partial JSON
			if block.Type == anthropic.MessagesContentTypeToolUse &&
				block.MessageContentToolUse != nil {
				tc := ToolCall{
					ID:   block.MessageContentToolUse.ID,
					Type: "function",
					Function: ToolCallFunction{
						Name:      block.MessageContentToolUse.Name,
						Arguments: string(block.MessageContentToolUse.Input),
					},
				}
				toolCalls = append(toolCalls, tc)

				// Now send partial update with the new tool call
				eventsChan <- ChatCompletionResponse{
					Choices: []Choice{{
						Index: 0,
						Message: OutputMessage{
							Role:      RoleAssistant,
							Content:   partialTextBuilder.String(),
							ToolCalls: toolCalls,
						},
						FinishReason: FinishReasonNull,
					}},
				}
			}
		},

		OnMessageDelta: func(d anthropic.MessagesEventMessageDeltaData) {
			// This might indicate a new stop reason or usage changes
			// if d.Delta.StopReason != "" {
			// 	stopReason = d.Delta.StopReason
			// }
			// We ignore usage here, or we could track usage tokens
		},

		OnMessageStop: func(d anthropic.MessagesEventMessageStopData) {
			// This indicates the final chunk of the stream for the message
			// We'll push a final chunk with the full content
			// finalMsg := ChatCompletionResponse{
			// 	Choices: []Choice{{
			// 		Index: 0,
			// 		Message: OutputMessage{
			// 			Role:      RoleAssistant,
			// 			Content:   partialTextBuilder.String(),
			// 			ToolCalls: toolCalls,
			// 		},
			// 		FinishReason: convertFromClaudeFinishReason(stopReason),
			// 	}},
			// }
			// eventsChan <- finalMsg

			// streaming is complete, so close the channel
			wrapper.Close()
		},
	}

	// Run the streaming request in a goroutine
	go func() {
		defer func() {
			// Ensure we close everything if an unexpected error occurs
			_ = wrapper.Close()
		}()

		_, err := c.client.CreateMessagesStream(ctxStream, streamReq)
		if err != nil && !errors.Is(err, io.EOF) {
			select {
			case errChan <- err:
			default:
			}
		}
	}()

	return wrapper, nil
}
