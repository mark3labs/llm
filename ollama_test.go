package llm

import (
	"context"
	"os"
	"strings"
	"testing"
)

const (
	testModel = "llama3.2:3b"
)

func getOllamaClient(t *testing.T) *OllamaLLM {
	// Allow overriding the Ollama host through env var
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://localhost:11434" // default Ollama host
	}

	client, err := NewOllamaLLM(host)
	if err != nil {
		t.Fatalf("Failed to create Ollama client: %v", err)
	}
	return client
}

func TestOllamaIntegration_BasicCompletion(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	client := getOllamaClient(t)
	ctx := context.Background()

	req := ChatCompletionRequest{
		Model: Model(testModel),
		Messages: []InputMessage{
			{
				Role: RoleUser,
				MultiContent: []ContentPart{
					{
						Type: ContentTypeText,
						Text: "What is 2+2?",
					},
				},
			},
		},
		Temperature: 0.0, // Use deterministic output
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		t.Fatalf("CreateChatCompletion failed: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("Expected at least one choice in response")
	}
	if resp.Choices[0].Message.Content == "" {
		t.Fatal("Expected non-empty message content")
	}
	if resp.Usage.TotalTokens == 0 {
		t.Fatal("Expected non-zero token usage")
	}
}

func TestOllamaIntegration_SystemPrompt(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	client := getOllamaClient(t)
	ctx := context.Background()

	systemPrompt := "You are a helpful math tutor who always shows your work."
	req := ChatCompletionRequest{
		Model:        Model(testModel),
		SystemPrompt: &systemPrompt,
		Messages: []InputMessage{
			{
				Role: RoleUser,
				MultiContent: []ContentPart{
					{
						Type: ContentTypeText,
						Text: "What is 15 * 7?",
					},
				},
			},
		},
		Temperature: 0.0,
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		t.Fatalf("CreateChatCompletion failed: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("Expected at least one choice in response")
	}
	if resp.Choices[0].Message.Content == "" {
		t.Fatal("Expected non-empty message content")
	}
	if !strings.Contains(resp.Choices[0].Message.Content, "105") {
		t.Error("Expected response to contain the answer '105'")
	}
}

func TestOllamaIntegration_Streaming(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	client := getOllamaClient(t)
	ctx := context.Background()

	req := ChatCompletionRequest{
		Model: Model(testModel),
		Messages: []InputMessage{
			{
				Role: RoleUser,
				MultiContent: []ContentPart{
					{
						Type: ContentTypeText,
						Text: "Count from 1 to 5.",
					},
				},
			},
		},
		Temperature: 0.0,
	}

	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		t.Fatalf("CreateChatCompletionStream failed: %v", err)
	}
	defer stream.Close()

	var receivedChunks []string
	for {
		resp, err := stream.Recv()
		if err != nil {
			break
		}
		if len(resp.Choices) > 0 && resp.Choices[0].Message.Content != "" {
			receivedChunks = append(receivedChunks, resp.Choices[0].Message.Content)
		}
	}

	if len(receivedChunks) == 0 {
		t.Fatal("Expected to receive streaming chunks")
	}
}

func TestOllamaIntegration_Conversation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	client := getOllamaClient(t)
	ctx := context.Background()

	req := ChatCompletionRequest{
		Model: Model(testModel),
		Messages: []InputMessage{
			{
				Role: RoleUser,
				MultiContent: []ContentPart{
					{
						Type: ContentTypeText,
						Text: "My name is Alice.",
					},
				},
			},
			{
				Role: RoleAssistant,
				MultiContent: []ContentPart{
					{
						Type: ContentTypeText,
						Text: "Hello Alice! Nice to meet you.",
					},
				},
			},
			{
				Role: RoleUser,
				MultiContent: []ContentPart{
					{
						Type: ContentTypeText,
						Text: "What's my name?",
					},
				},
			},
		},
		Temperature: 0.0,
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		t.Fatalf("CreateChatCompletion failed: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("Expected at least one choice in response")
	}
	if !strings.Contains(strings.ToLower(resp.Choices[0].Message.Content), "alice") {
		t.Error("Expected response to contain the name 'Alice'")
	}
}
