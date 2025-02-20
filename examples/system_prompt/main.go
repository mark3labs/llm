package main

import (
	"context"
	"fmt"

	"github.com/dataleap-labs/llm"
)

func main() {

	openAIKey := ""

	openai := llm.NewOpenAILLM(openAIKey)

	fmt.Println("OpenAI client initialized.")

	systemPrompt := "You only respond with Emojis"
	imageRequest := llm.ChatCompletionRequest{
		SystemPrompt: &systemPrompt,
		Model:        llm.ModelGPT4o,
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

	response, err := openai.CreateChatCompletion(context.Background(), imageRequest)
	if err != nil {
		panic(err)
	}

	fmt.Println("OpenAI: ", response.Choices[0].Message.Content, "Completion tokens: ", response.Usage.CompletionTokens, "Prompt tokens: ", response.Usage.PromptTokens, "Total tokens: ", response.Usage.TotalTokens)

}
