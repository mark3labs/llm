package llm

import "context"

// Role represents the role of a conversation participant.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// LLMProvider represents different LLM SDK providers.
type LLMProvider string

const (
	OpenAIProvider LLMProvider = "openai"
	GeminiProvider LLMProvider = "gemini"
	ClaudeProvider LLMProvider = "claude"
)

type Model string

const (
	ModelChatGPT4oLatest     Model = "chatgpt-4o-latest"
	ModelGPT4o               Model = "gpt-4o"
	ModelGPT4oMini           Model = "gpt-4o-mini"
	ModelGPT4o2024_08_06     Model = "gpt-4o-2024-08-06"
	ModelGPT4oMini2024_07_18 Model = "gpt-4o-mini-2024-07-18"
	ModelO1                  Model = "o1"
	ModelO1_2024_12_17       Model = "o1-2024-12-17"
	ModelO1Preview2024_09_12 Model = "o1-preview-2024-09-12"
	ModelO1Preview           Model = "o1-preview"
	ModelO1Mini              Model = "o1-mini"
	ModelO1Mini2024_09_12    Model = "o1-mini-2024-09-12"
	ModelO3Mini              Model = "o3-mini"
	ModelO3Mini2025_01_31    Model = "o3-mini-2025-01-31"

	ModelClaude2Dot0               Model = "claude-2.0"
	ModelClaude2Dot1               Model = "claude-2.1"
	ModelClaude3Opus20240229       Model = "claude-3-opus-20240229"
	ModelClaude3Sonnet20240229     Model = "claude-3-sonnet-20240229"
	ModelClaude3Dot5Sonnet20240620 Model = "claude-3-5-sonnet-20240620"
	ModelClaude3Dot5Sonnet20241022 Model = "claude-3-5-sonnet-20241022"
	ModelClaude3Dot5SonnetLatest   Model = "claude-3-5-sonnet-latest"
	ModelClaude3Haiku20240307      Model = "claude-3-haiku-20240307"
	ModelClaude3Dot5HaikuLatest    Model = "claude-3-5-haiku-latest"
	ModelClaude3Dot5Haiku20241022  Model = "claude-3-5-haiku-20241022"

	ModelGemini2Flash                Model = "gemini-2.0-flash"
	ModelGemini2FlashLite001        Model = "gemini-2.0-flash-lite-001"
	ModelGemini15Flash               Model = "gemini-1.5-flash"
	ModelGemini15Flash8B             Model = "gemini-1.5-flash-8b"
	ModelGemini15Pro                 Model = "gemini-1.5-pro"
)

type ContentPart struct {
	Type      ContentType
	Text      string
	Data      string
	MediaType string
}

type ContentType string

const (
	ContentTypeText  ContentType = "text"  // ContentTypeText indicates that a content part is text.
	ContentTypeImage ContentType = "image" // ContentTypeImage indicates that a content part is an image.
)

// Message represents a single message in a conversation.
type InputMessage struct {
	Role         Role          `json:"role"`
	MultiContent []ContentPart `json:"content,omitempty"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolResults  []ToolResult  `json:"tool_results,omitempty"`
}

type OutputMessage struct {
	Role      Role       `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ChatCompletionRequest represents a request for a chat completion.
type ChatCompletionRequest struct {
	Model        Model          `json:"model"`
	Messages     []InputMessage `json:"messages"`
	SystemPrompt *string        `json:"system_prompt,omitempty"`
	Tools        []Tool         `json:"tools,omitempty"`
	Temperature  float32        `json:"temperature,omitempty"`
	TopP         *float32       `json:"top_p,omitempty"`
	MaxTokens    int            `json:"max_tokens,omitempty"`
	JSONMode     bool           `json:"json_mode,omitempty"`
}

// Tool represents a function that can be called by the LLM
type Tool struct {
	Type     string    `json:"type"`
	Function *Function `json:"function,omitempty"`
}

type ToolResult struct {
	ToolCallID   string
	FunctionName string
	Result       string
	IsError      bool
}

// Function represents a function definition
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ToolCall represents a tool/function call from the LLM
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatCompletionResponse represents the response from a chat completion request.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type FinishReason string

const (
	FinishReasonToolCalls FinishReason = "tool_calls"
	FinishReasonStop      FinishReason = "stop"
	FinishReasonMaxTokens FinishReason = "max_tokens"
	FinishReasonNull      FinishReason = "null"
)

// Choice represents a single completion choice.
type Choice struct {
	Index        int           `json:"index"`
	Message      OutputMessage `json:"message"`
	FinishReason FinishReason  `json:"finish_reason"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// LLM defines the interface that all LLM providers must implement.
type LLM interface {
	CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error)
	CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (ChatCompletionStream, error)
}

// ChatCompletionStream represents a streaming chat completion.
type ChatCompletionStream interface {
	Recv() (ChatCompletionResponse, error)
	Close() error
}
