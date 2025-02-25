package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// GeminiLLM implements the LLM interface for Google's Gemini
type GeminiLLM struct {
	client *genai.Client
}

// GeminiOptions contains configuration options for the Gemini model
type GeminiOptions struct {
	Model          string
	HarmThreshold  genai.HarmBlockThreshold
	SafetySettings []*genai.SafetySetting
}

// NewGeminiLLM creates a new Gemini LLM client
func NewGeminiLLM(apiKey string, opts ...GeminiOptions) (*GeminiLLM, error) {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %v", err)
	}

	return &GeminiLLM{
		client: client,
	}, nil
}

// convertToGeminiMessages converts our generic Message type to Gemini's content type
func convertToGeminiMessages(messages []InputMessage) []genai.Content {
	var contents []genai.Content

	for _, msg := range messages {
		parts := convertToGeminiParts(msg.MultiContent)
		var content genai.Content

		switch msg.Role {
		case RoleTool:
			// For tool results, treat them as user content with a function response
			if len(msg.ToolResults) > 0 {
				tr := msg.ToolResults[0]
				response := map[string]any{
					"response": map[string]any{
						"name":    tr.FunctionName,
						"content": tr.Result,
					},
				}
				parts = append(parts, genai.FunctionResponse{
					Name:     tr.ToolCallID,
					Response: response,
				})
			}
			content.Role = "user"
		case RoleAssistant:
			content.Role = "model"
		case RoleUser:
			content.Role = "user"
		}

		content.Parts = parts

		// If the assistant had a tool call
		if msg.Role == RoleAssistant && len(msg.ToolCalls) > 0 {
			// We'll store them as if the assistant invoked a function
			for _, tc := range msg.ToolCalls {
				argsJSON := make(map[string]any)
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &argsJSON)
				call := genai.FunctionCall{
					Name: tc.Function.Name,
					Args: argsJSON,
				}
				parts = append(parts, call)
			}
			content.Parts = parts
		}

		contents = append(contents, content)
	}

	return contents
}

func convertToGeminiParts(content []ContentPart) []genai.Part {
	multiContent := make([]genai.Part, 0, len(content))
	for _, part := range content {
		switch part.Type {
		case ContentTypeText:
			multiContent = append(multiContent, genai.Text(part.Text))
		case ContentTypeImage:
			imageBytes, err := base64.StdEncoding.DecodeString(part.Data)
			if err != nil {
				continue // Skip if decoding fails
			}
			multiContent = append(multiContent, genai.Blob{
				Data:     imageBytes,
				MIMEType: part.MediaType,
			})
		}
	}
	return multiContent
}

// convertToGeminiTools converts our generic Tool type to Gemini's tool type
func convertToGeminiTools(tools []Tool) []*genai.Tool {
	if len(tools) == 0 {
		return nil
	}

	geminiTools := make([]*genai.Tool, len(tools))
	for i, tool := range tools {
		schema := &genai.Schema{
			Type: genai.TypeObject,
		}

		schema.Properties = make(map[string]*genai.Schema)

		if properties, ok := tool.Function.Parameters["properties"].(map[string]interface{}); ok {
			for name, prop := range properties {
				if propMap, ok := prop.(map[string]interface{}); ok {
					propSchema := &genai.Schema{}
					if typ, ok := propMap["type"].(string); ok {
						propSchema.Type = convertSchemaType(typ)
					}
					if desc, ok := propMap["description"].(string); ok {
						propSchema.Description = desc
					}
					schema.Properties[name] = propSchema
				}
			}
		}

		if required, ok := tool.Function.Parameters["required"].([]interface{}); ok {
			reqFields := make([]string, len(required))
			for i, r := range required {
				if str, ok := r.(string); ok {
					reqFields[i] = str
				}
			}
			schema.Required = reqFields
		}

		geminiTools[i] = &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  schema,
				},
			},
		}
	}
	return geminiTools
}

// convertSchemaType converts a JSON Schema type to Gemini schema type
func convertSchemaType(typ string) genai.Type {
	switch typ {
	case "object":
		return genai.TypeObject
	case "string":
		return genai.TypeString
	case "number":
		return genai.TypeNumber
	case "integer":
		return genai.TypeInteger
	case "boolean":
		return genai.TypeBoolean
	case "array":
		return genai.TypeArray
	default:
		return genai.TypeUnspecified
	}
}

// convertFromGeminiToolCalls (unused in streaming approach, for reference)
func convertFromGeminiToolCalls(parts []genai.Part) []ToolCall {
	var calls []ToolCall
	for _, part := range parts {
		if fc, ok := part.(genai.FunctionCall); ok {
			args, _ := json.Marshal(fc.Args)
			calls = append(calls, ToolCall{
				Type: "function",
				Function: ToolCallFunction{
					Name:      fc.Name,
					Arguments: string(args),
				},
			})
		}
	}
	return calls
}

// CreateChatCompletion implements the LLM interface for Gemini (non-streaming).
func (g *GeminiLLM) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	if !g.isSupported(req.Model) {
		return ChatCompletionResponse{}, fmt.Errorf("model %s is not supported", req.Model)
	}

	modelName := string(req.Model)
	model := g.client.GenerativeModel(modelName)

	// Set system prompt if provided
	if req.SystemPrompt != nil {
		model.SystemInstruction = &genai.Content{
			Parts: []genai.Part{
				genai.Text(*req.SystemPrompt),
			},
		}
	}

	setModelConfig(model, req)

	// Convert messages to Gemini format
	geminiMessages := convertToGeminiMessages(req.Messages)

	// Gemini requires at least one message
	if len(geminiMessages) == 0 {
		return ChatCompletionResponse{}, fmt.Errorf("no messages provided")
	}

	chatSession := model.StartChat()
	loadChatSession(chatSession, geminiMessages[:len(geminiMessages)-1])
	newMessage := geminiMessages[len(geminiMessages)-1]
	resp, err := chatSession.SendMessage(ctx, newMessage.Parts...)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("failed to generate content: %v", err)
	}

	// Convert response to our format
	choices := make([]Choice, len(resp.Candidates))
	for i, c := range resp.Candidates {
		choices[i] = convertFromGeminiCandidate(c, i)
	}

	response := ChatCompletionResponse{
		Choices: choices,
	}

	if resp.UsageMetadata != nil {
		response.Usage = Usage{
			PromptTokens:     int(resp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(resp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      int(resp.UsageMetadata.TotalTokenCount),
		}
	}

	return response, nil
}

func convertFromGeminiCandidate(c *genai.Candidate, index int) Choice {
	msg := OutputMessage{
		Role:    RoleAssistant,
		Content: "",
	}
	var textParts []string
	for _, part := range c.Content.Parts {
		switch p := part.(type) {
		case genai.Text:
			textParts = append(textParts, string(p))
		case genai.FunctionCall:
			args, err := json.Marshal(p.Args)
			if err != nil {
				continue
			}
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				Type: "function",
				Function: ToolCallFunction{
					Name:      p.Name,
					Arguments: string(args),
				},
			})
		}
	}
	msg.Content = strings.Join(textParts, "")

	return Choice{
		Index:        index,
		Message:      msg,
		FinishReason: FinishReason(c.FinishReason),
	}
}

func setModelConfig(model *genai.GenerativeModel, req ChatCompletionRequest) {

	// https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values
	// Default value is not 0. It's safer to set the temperature to a small non zero value to avoid the initial value from being lost when marshalled/unmarshalled when sending over an API
	if req.Temperature > 0 {
		model.SetTemperature(float32(req.Temperature))
	} else {
		model.SetTemperature(math.SmallestNonzeroFloat32)
	}

	if req.TopP != nil && *req.TopP > 0 {
		model.SetTopP(*req.TopP)
	}

	model.SetMaxOutputTokens(int32(req.MaxTokens))

	if req.JSONMode {
		model.ResponseMIMEType = "application/json"
	}

	geminiTools := convertToGeminiTools(req.Tools)
	model.Tools = geminiTools
}

// isSupported checks if the given model is recognized as a valid Gemini model
func (g *GeminiLLM) isSupported(model Model) bool {
	switch model {
	case ModelGemini2Flash, ModelGemini15Flash, ModelGemini15Flash8B, ModelGemini15Pro, ModelGemini2FlashLite001:
		return true
	default:
		return false
	}
}

// geminiStreamWrapper wraps Gemini's streaming iterator to implement our ChatCompletionStream interface
type geminiStreamWrapper struct {
	iter                 *genai.GenerateContentResponseIterator
	done                 bool
	accumulatedText      string     // aggregator for text so far
	accumulatedToolCalls []ToolCall // aggregator for tool calls so far
}

// Recv returns the next partial or final ChatCompletionResponse from Gemini.
func (w *geminiStreamWrapper) Recv() (ChatCompletionResponse, error) {
	if w.done {
		return ChatCompletionResponse{}, io.EOF
	}

	resp, err := w.iter.Next()
	if err != nil {
		if errors.Is(err, iterator.Done) {
			return ChatCompletionResponse{}, io.EOF
		}
		return ChatCompletionResponse{}, err
	}

	if len(resp.Candidates) == 0 {
		// If no candidates, return an empty partial update
		return ChatCompletionResponse{
			Choices: []Choice{{
				Index: 0,
				Message: OutputMessage{
					Role:    RoleAssistant,
					Content: "",
				},
				FinishReason: FinishReasonNull,
			}},
		}, nil
	}

	// We'll only handle the first candidate for partial streaming.
	candidate := resp.Candidates[0]
	var newText string
	var newToolCalls []ToolCall

	// 1. Extract text/tool calls from this partial
	for _, part := range candidate.Content.Parts {
		switch p := part.(type) {
		case genai.Text:
			newText += string(p)
		case genai.FunctionCall:
			args, e := json.Marshal(p.Args)
			if e == nil {
				newToolCalls = append(newToolCalls, ToolCall{
					Type: "function",
					Function: ToolCallFunction{
						Name:      p.Name,
						Arguments: string(args),
					},
				})
			}
		}
	}

	// 2. Convert new text into partial delta
	var deltaContent string
	oldLen := len(w.accumulatedText)
	w.accumulatedText += newText
	if len(w.accumulatedText) > oldLen {
		deltaContent = w.accumulatedText[oldLen:]
	}

	// 3. Convert new tool calls into partial (any calls that did not appear before)
	var deltaCalls []ToolCall
	for _, tc := range newToolCalls {
		// naive approach: if not already in accumulatedToolCalls, then it's new
		isNew := true
		for _, existing := range w.accumulatedToolCalls {
			if existing.Function.Name == tc.Function.Name &&
				existing.Function.Arguments == tc.Function.Arguments {
				isNew = false
				break
			}
		}
		if isNew {
			deltaCalls = append(deltaCalls, tc)
			w.accumulatedToolCalls = append(w.accumulatedToolCalls, tc)
		}
	}

	// 4. Determine finish reason
	fr := FinishReasonNull
	switch candidate.FinishReason {
	case genai.FinishReasonStop:
		if len(w.accumulatedToolCalls) > 0 {
			fr = FinishReasonToolCalls
			w.done = true
		} else {
			fr = FinishReasonStop
			w.done = true
		}
	case genai.FinishReasonSafety:
		// The gemini library might block or produce partial or final
		// We'll treat those like a stop with an error or just "stop"
		fr = FinishReasonStop
		w.done = true
	case genai.FinishReasonMaxTokens:
		fr = FinishReasonMaxTokens
		w.done = true
	case genai.FinishReasonUnspecified:
		fr = FinishReasonNull
		w.done = false
	default:
		err = fmt.Errorf("unknown finish reason: %v", candidate.FinishReason)
		panic(err)
	}

	// 5. Construct the partial chunk response
	chunk := ChatCompletionResponse{
		Choices: []Choice{
			{
				Index: 0,
				Message: OutputMessage{
					Role:      RoleAssistant,
					Content:   deltaContent,
					ToolCalls: deltaCalls,
				},
				FinishReason: fr,
			},
		},
	}

	return chunk, nil
}

// Close signals we are finished with the stream
func (w *geminiStreamWrapper) Close() error {
	// If there's an open gRPC or HTTP connection it is closed automatically after iteration
	// but to align with the interface, we can set iter to nil
	w.iter = nil
	w.done = true
	return nil
}

// CreateChatCompletionStream implements the LLM interface for Gemini streaming
func (g *GeminiLLM) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (ChatCompletionStream, error) {
	if !g.isSupported(req.Model) {
		return nil, fmt.Errorf("model %s is not supported", req.Model)
	}

	modelName := string(req.Model)
	model := g.client.GenerativeModel(modelName)

	setModelConfig(model, req)

	// Convert messages to Gemini format
	geminiMessages := convertToGeminiMessages(req.Messages)
	if len(geminiMessages) == 0 {
		return nil, fmt.Errorf("no messages provided")
	}

	chatSession := model.StartChat()
	loadChatSession(chatSession, geminiMessages[:len(geminiMessages)-1])
	newMessage := geminiMessages[len(geminiMessages)-1]
	respIter := chatSession.SendMessageStream(ctx, newMessage.Parts...)

	return &geminiStreamWrapper{
		iter: respIter,
	}, nil
}

func loadChatSession(chatSession *genai.ChatSession, geminiMessages []genai.Content) {
	if len(geminiMessages) > 1 {
		historyPtr := make([]*genai.Content, len(geminiMessages))
		for i := 0; i < len(geminiMessages); i++ {
			historyPtr[i] = &geminiMessages[i]
		}
		chatSession.History = historyPtr
	}
}
