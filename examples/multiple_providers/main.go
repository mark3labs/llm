package main

import (
	"context"
	"fmt"
	"os"

	"github.com/dataleap-labs/llm"
)

func main() {

	openAIKey := ""
	anthropicKey := ""
	geminiKey := ""

	gemini, err := llm.NewGeminiLLM(geminiKey)
	if err != nil {
		panic(err)
	}

	// read google credentials from file
	credBytes, err := os.ReadFile("google.json")
	if err != nil {
		panic(err)
	}

	claude := llm.NewAnthropicLLM(anthropicKey)
	claudeVertex := llm.NewVertexLLM(credBytes, "project-id", "location")

	openai := llm.NewOpenAILLM(openAIKey)

	fmt.Println("Gemini initialized.")
	fmt.Println("Claude initialized.")
	fmt.Println("OpenAI initialized.")

	imageRequest := llm.ChatCompletionRequest{
		Messages: []llm.InputMessage{
			{
				Role: llm.RoleUser,
				MultiContent: []llm.ContentPart{
					{
						Type: llm.ContentTypeText,
						Text: "What is the purpose of life?",
					},
				},
			},
		},
		JSONMode:    false,
		MaxTokens:   1000,
		Temperature: 0,
	}

	imageRequest.Model = llm.ModelClaude3Dot5Sonnet20241022

	response, err := claude.CreateChatCompletion(context.Background(), imageRequest)
	if err != nil {
		panic(err)
	}
	fmt.Println("Claude: ", response.Choices[0].Message.Content, "Completion tokens: ", response.Usage.CompletionTokens, "Prompt tokens: ", response.Usage.PromptTokens, "Total tokens: ", response.Usage.TotalTokens)

	response, err = claudeVertex.CreateChatCompletion(context.Background(), imageRequest)
	if err != nil {
		panic(err)
	}
	fmt.Println("Claude Vertex: ", response.Choices[0].Message.Content, "Completion tokens: ", response.Usage.CompletionTokens, "Prompt tokens: ", response.Usage.PromptTokens, "Total tokens: ", response.Usage.TotalTokens)

	imageRequest.Model = llm.ModelGPT4o

	response, err = openai.CreateChatCompletion(context.Background(), imageRequest)
	if err != nil {
		panic(err)
	}

	fmt.Println("OpenAI: ", response.Choices[0].Message.Content, "Completion tokens: ", response.Usage.CompletionTokens, "Prompt tokens: ", response.Usage.PromptTokens, "Total tokens: ", response.Usage.TotalTokens)

	imageRequest.Model = llm.ModelGemini15Flash8B

	response, err = gemini.CreateChatCompletion(context.Background(), imageRequest)
	if err != nil {
		panic(err)
	}

	fmt.Println("Gemini: ", response.Choices[0].Message.Content, "Completion tokens: ", response.Usage.CompletionTokens, "Prompt tokens: ", response.Usage.PromptTokens, "Total tokens: ", response.Usage.TotalTokens)

}
