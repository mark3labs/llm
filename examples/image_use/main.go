package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"

	"github.com/dataleaplabs/llm"
)

func main() {

	openAIKey := ""

	openai := llm.NewOpenAILLM(openAIKey)

	fmt.Println("OpenAI client initialized.")

	// read image from test_images/cats.png and convert to base64
	imageBytesDogs, err := os.ReadFile("images/dogs.png")
	if err != nil {
		panic(err)
	}

	imageBytesCats, err := os.ReadFile("images/cats.png")
	if err != nil {
		panic(err)
	}

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
						Text: `Compare the two images.
						`,
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
