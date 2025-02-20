package llm

import (
	"context"
	"io"
	"strings"
)

// StreamHandler defines how to handle streaming tokens, tool calls,
// and completion events from an LLM.
type StreamHandler interface {
	// Called once right before tokens start streaming.
	OnStart()

	// Called whenever the LLM produces a new token (partial output).
	OnToken(token string)

	// If the LLM triggers an external tool call, you can capture it here.
	OnToolCall(toolCall ToolCall)

	// Called when the LLM produces a final complete message.
	OnComplete(message OutputMessage)

	// Called if an error occurs during the streaming process.
	OnError(err error)
}

func StreamChatCompletion(
	ctx context.Context,
	req ChatCompletionRequest,
	handler StreamHandler,
	model LLM,
) error {
	stream, err := model.CreateChatCompletionStream(ctx, req)
	if err != nil {
		handler.OnError(err)
		return err
	}

	handler.OnStart()

	var fullContent strings.Builder
	var toolCalls []ToolCall
	defer func() {
		// In case you need to close the stream
		_ = stream.Close()
	}()

	for {
		chunk, err := stream.Recv() // however you read from your streaming LLM
		if err != nil {
			if isEOF(err) {
				// Done reading
				break
			}
			handler.OnError(err)
			return err
		}

		// 	// Usually the chunk includes tokens. For example:
		for _, c := range chunk.Choices {

			// If there's a partial delta (like with OpenAI's usage of .Delta)
			if len(c.Message.Content) > 0 {
				handler.OnToken(c.Message.Content)
				fullContent.WriteString(c.Message.Content)
			}

			if len(c.Message.ToolCalls) > 0 {
				toolCalls = append(toolCalls, c.Message.ToolCalls...)
			}

			// If there's a tool call signaled
			if c.FinishReason == FinishReasonToolCalls {
				// pass in your llm.ToolCall
				lastToolCall := toolCalls[len(toolCalls)-1]
				handler.OnToolCall(lastToolCall)
			}
			// If there's a final completion event
			if c.FinishReason != FinishReasonNull {
				// We got the final message, call OnComplete with the final message
				msg := OutputMessage{
					Role:      "assistant",
					Content:   fullContent.String(),
					ToolCalls: toolCalls,
				}
				handler.OnComplete(msg)
				return nil
			}
		}
	}

	return nil
}

func isEOF(err error) bool {
	return err == io.EOF
}
