The purpose of this project is to provide a common interface for LLMs (Sonnet, GPT4o etc) across different API providers (Anthropic, Bedrock, Vertex, OpenAI, ...).

Currently we support the following LLMs and API providers:

- Claude models on Anthropic and Vertex (Google)
- OpenAI models on OpenAI and Azure
- Gemini models on Vertex

## Examples

<details>
<summary>Simple text message</summary>

```
openAIKey := "abc"
openai := llm.NewOpenAILLM(openAIKey)

systemPrompt := "You only use Emojis."
textRequest := llm.ChatCompletionRequest{
		SystemPrompt: &systemPrompt,
		Model:        llm.ModelGPT4o,
		Messages: []llm.InputMessage{
			{
				Role: llm.RoleUser,
				MultiContent: []llm.ContentPart{
					{
						Type: llm.ContentTypeText,
						Text: `What is the purpose of life?`
					},
				},
			},
		},
		JSONMode:    false,
		MaxTokens:   1000,
		Temperature: 0,
	}

response, err := openai.CreateChatCompletion(context.Background(), textRequest)
if err != nil {
	panic(err)
}

fmt.Println(response)
```

</details>

<details>
<summary>Initialize Claude, Gemini and OpenAI client</summary>

```
// OpenAI model from OpenAI API
openAIKey := ""
openai := llm.NewOpenAILLM(openAIKey)

// Claude model from OpenAI API
anthropicKey := ""
claude := llm.NewAnthropicLLM(anthropicKey)

// Gemini model from Vertex API
geminiKey := ""
gemini, err := llm.NewGeminiLLM(geminiKey)
if err != nil {
    panic(err)
}

// Claude model from Vertex API
// read google credentials from file
credBytes, err := os.ReadFile("google.json")
if err != nil {
    panic(err)
}

claudeVertex := llm.NewVertexLLM(credBytes, "project-id", "location")
```

</details>

<details>
<summary>Message with image input</summary>

```
openAIKey := ""
openai := llm.NewOpenAILLM(openAIKey)


// read image from test_images/cats.png and convert to base64
imageBytesDogs, _ := os.ReadFile("images/dogs.png")
imageBytesCats, _ := os.ReadFile("images/cats.png")


imageBase64Dogs := base64.StdEncoding.EncodeToString(imageBytesDogs)
imageBase64Cats := base64.StdEncoding.EncodeToString(imageBytesCats)

systemPrompt := "You are a cat and speak in cat language."
imageRequest := llm.ChatCompletionRequest{
	SystemPrompt: &systemPrompt,
	Model:        llm.ModelGPT4o,
	Messages: []llm.InputMessage{
		{
			Role: llm.RoleUser,
			MultiContent: []llm.ContentPart{
				{
					Type: llm.ContentTypeText,
					Text: "Image number 1",
				},
				{
					Type:      llm.ContentTypeImage,
					Data:      imageBase64Cats,
					MediaType: "image/png",
				},
				{
					Type: llm.ContentTypeText,
					Text: "Image number 2",
				},
				{
					Type:      llm.ContentTypeImage,
					Data:      imageBase64Dogs,
					MediaType: "image/png",
				},
				{
					Type: llm.ContentTypeText,
					Text: `Compare the two images`.
				},
			},
		},
	},
	JSONMode:    false,
	MaxTokens:   1000,
	Temperature: 0,
}

response, _ := openai.CreateChatCompletion(context.Background(), imageRequest)
fmt.Println(reponse)
```

</details>

<details>
<summary>Message with tool use</summary>

```
openAIKey := ""
openai := llm.NewOpenAILLM(openAIKey)

toolRequestWithToolResponse := llm.ChatCompletionRequest{
	Model: llm.ModelGPT4o,
	Messages: []llm.InputMessage{
		{
			Role: llm.RoleUser,
			MultiContent: []llm.ContentPart{
				{
					Type: llm.ContentTypeText,
					Text: "What is the weather in Paris, France?",
				},
			},
		},
		{
			Role: llm.RoleAssistant,
			ToolCalls: []llm.ToolCall{{
				ID:   "123",
				Type: "function",
				Function: llm.ToolCallFunction{
					Name:      "get_weather",
					Arguments: "{\"location\": \"Paris, France\"}",
				},
			}},
		},
		{
			Role:         llm.RoleTool,
			MultiContent: nil,
			ToolResults: []llm.ToolResult{
				{
					ToolCallID:   "123",
					FunctionName: "get_weather",
					Result:       "15 degrees",
					IsError:      false,
				},
			},
		},
	},
	Tools: []llm.Tool{
		{
			Type: "function",
			Function: &llm.Function{
				Name:        "get_weather",
				Description: "Get current temperature for a given location.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "City and country e.g. Bogotá, Colombia",
						},
					},
					"required":             []string{"location"},
					"additionalProperties": false,
				},
			},
		},
	},
	JSONMode:    false,
	MaxTokens:   1000,
	Temperature: 0,
}

response, _ := openai.CreateChatCompletion(context.Background(), toolRequestWithToolResponse)
fmt.Println(response)
```

</details>

## Overview

The package structure is quite simple. There are two core interfaces. The first interface is the LLM interface which is implemented for every LLM model. E.g. in the file claude.go you can see the implementation of that interface for the claude models.

```
type LLM interface {
	CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error)
	CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (ChatCompletionStream, error)
}
```

The second interface is the StreamHandler interface. You can use it to define your own stream handler e.g. for SSE streaming over a RestAPI. In the example folder `streaming` you can find a simple implementation which just prints out incoming tokens.

```
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
```

## Roadmap

- Accurate token counting across models
- Tests
- Contribute bedrock adapter in `https://github.com/liushuangls/go-anthropic` to also allow us to use it here
- Add support for computer use
- Better error messages for features that are only supported for some models or API providers (e.g. caching)
- Batch processing

## Credits

OpenAI model integration
https://github.com/sashabaranov/go-openai

Antropic model integration
https://github.com/liushuangls/go-anthropic
