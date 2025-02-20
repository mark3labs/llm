package main

import (
	"context"
	"fmt"

	"github.com/dataleap-labs/llm"
)

func main() {

	openAIKey := ""

	openai := llm.NewOpenAILLM(openAIKey)

	fmt.Println("OpenAI initialized")

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

	response, err := openai.CreateChatCompletion(context.Background(), toolRequestWithToolResponse)
	if err != nil {
		panic(err)
	}

	fmt.Println("OpenAI: ", response.Choices[0].Message.Content, "Completion tokens: ", response.Usage.CompletionTokens, "Prompt tokens: ", response.Usage.PromptTokens, "Total tokens: ", response.Usage.TotalTokens)
}
