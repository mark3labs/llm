package main

import (
	"fmt"
	"strings"

	"github.com/dataleap-labs/llm"
)

// Ensure SimpleStreamHandler implements the StreamHandler interface.
var _ llm.StreamHandler = (*SimpleStreamHandler)(nil)

func NewSimpleStreamHandler() *SimpleStreamHandler {
	return &SimpleStreamHandler{
		builder: &strings.Builder{},
	}
}

type SimpleStreamHandler struct {
	builder *strings.Builder
}

// OnStart resets any internal state needed for a new stream.
func (h *SimpleStreamHandler) OnStart() {
	fmt.Println("[SimpleStreamHandler] Stream started.")
	h.builder.Reset()
}

// OnToken appends new tokens to an internal string builder.
func (h *SimpleStreamHandler) OnToken(token string) {
	fmt.Printf("[SimpleStreamHandler] Received token: %s\n", token)
	h.builder.WriteString(token)
}

// OnToolCall logs the tool call and does nothing else in this simple example.
func (h *SimpleStreamHandler) OnToolCall(toolCall llm.ToolCall) {
	fmt.Printf("[SimpleStreamHandler] Tool call triggered: %+v\n", toolCall)
}

// OnComplete logs the final message and the aggregated output from the builder.
func (h *SimpleStreamHandler) OnComplete(message llm.OutputMessage) {
	fmt.Printf("[SimpleStreamHandler] Stream complete. Final message: %+v\n", message)
	// fmt.Printf("[SimpleStreamHandler] All tokens aggregated: %s\n", h.builder.String())
}

// OnError logs any error that occurred during streaming.
func (h *SimpleStreamHandler) OnError(err error) {
	fmt.Printf("[SimpleStreamHandler] Error: %v\n", err)
}
