package main

import (
	"context"

	"github.com/dataleaplabs/llm"
)

func main() {

	openAIKey := ""
	openai := llm.NewOpenAILLM(openAIKey)
	streamHandler := NewSimpleStreamHandler()

	streamingRequest := llm.ChatCompletionRequest{
		Model: llm.ModelGPT4o,
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

	err := llm.StreamChatCompletion(context.Background(), streamingRequest, streamHandler, openai)

	if err != nil {
		panic(err)
	}

}
