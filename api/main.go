package handler

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
)

type OpenAIRequest struct {
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Model    string    `json:"model"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// UnmarshalJSON 自定义JSON解析方法
func (m *Message) UnmarshalJSON(data []byte) error {
	type Alias Message
	aux := &struct {
		Role    string      `json:"role"`
		Content interface{} `json:"content"`
	}{}
	
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	
	m.Role = aux.Role
	
	// 处理content字段
	switch v := aux.Content.(type) {
	case string:
		m.Content = v
	case []interface{}:
		// 如果是数组，将所有元素连接成字符串
		var contents []string
		for _, item := range v {
			if str, ok := item.(string); ok {
				contents = append(contents, str)
			}
		}
		m.Content = strings.Join(contents, "")
	case nil:
		m.Content = ""
	default:
		m.Content = fmt.Sprintf("%v", v)
	}
	
	return nil
}

type OpenAIResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message,omitempty"`
	Delta        Message `json:"delta,omitempty"`
	FinishReason string  `json:"finish_reason"`
}

var modelMapping = map[string]string{
	"deepseek":          "deepseek/deepseek-chat",
	"gpt-4o-mini":       "openai/gpt-4o-mini",
	"gemini-flash-1.5":  "google/gemini-flash-1.5",
	"deepseek-reasoner": "deepseek-reasoner",
	"minimax-01":        "minimax/minimax-01",
}

func Handler(w http.ResponseWriter, r *http.Request) {
	// 设置CORS头部
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	
	// 处理OPTIONS请求
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if !strings.HasSuffix(r.URL.Path, "/v1/chat/completions") {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "Service Running",
			"message": "MoLoveSze...",
		})
		return
	}

	validToken := os.Getenv("AUTH_TOKEN")
	requestToken := r.Header.Get("Authorization")
	if validToken != "" {
		if requestToken != "Bearer "+validToken {
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]string{
				"error": "Unauthorized Access",
			})
			return
		}
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	var req OpenAIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fmt.Printf("Request Format Error: %v\n", err)
		http.Error(w, fmt.Sprintf("Request Format Error: %v", err), http.StatusBadRequest)
		return
	}

	actualModel, exists := modelMapping[req.Model]
	if !exists {
		fmt.Printf("Unsupported Model: %s\n", req.Model)
		http.Error(w, fmt.Sprintf("Unsupported Model: %s", req.Model), http.StatusBadRequest)
		return
	}

	var question string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			question = req.Messages[i].Content
			break
		}
	}

	sessionID := uuid.New().String()

	// 设置通用响应头
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		handleStreamResponse(w, question, sessionID, req.Messages, req.Model, actualModel)
	} else {
		w.Header().Set("Content-Type", "application/json")
		handleNormalResponse(w, question, sessionID, req.Messages, req.Model, actualModel)
	}
}

func handleStreamResponse(w http.ResponseWriter, question, sessionID string, messages []Message, requestModel, actualModel string) {
	// 设置SSE相关的响应头
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	resp := makeHeckRequest(question, sessionID, messages, actualModel)
	if resp.StatusCode != http.StatusOK {
		http.Error(w, "Upstream Service Error", http.StatusInternalServerError)
		return
	}

	defer resp.Body.Close()
	reader := bufio.NewReader(resp.Body)

	isAnswering := false
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}

		if strings.HasPrefix(line, "data: ") {
			content := strings.TrimPrefix(line, "data: ")
			content = strings.TrimSpace(content)

			if content == "[ANSWER_START]" {
				isAnswering = true
				chunk := OpenAIResponse{
					ID:      sessionID,
					Object:  "chat.completion.chunk",
					Created: time.Now().Unix(),
					Model:   requestModel,
					Choices: []Choice{
						{
							Index: 0,
							Delta: Message{
								Role: "assistant",
							},
						},
					},
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				continue
			}

			if content == "[ANSWER_DONE]" {
				chunk := OpenAIResponse{
					ID:      sessionID,
					Object:  "chat.completion.chunk",
					Created: time.Now().Unix(),
					Model:   requestModel,
					Choices: []Choice{
						{
							Index:        0,
							Delta:        Message{},
							FinishReason: "stop",
						},
					},
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				break
			}

			if isAnswering && content != "" &&
				!strings.HasPrefix(content, "[RELATE_Q_START]") &&
				!strings.HasPrefix(content, "[RELATE_Q_DONE]") {
				chunk := OpenAIResponse{
					ID:      sessionID,
					Object:  "chat.completion.chunk",
					Created: time.Now().Unix(),
					Model:   requestModel,
					Choices: []Choice{
						{
							Index: 0,
							Delta: Message{
								Content: content,
							},
						},
					},
				}
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
			}
		}
	}
}

func handleNormalResponse(w http.ResponseWriter, question, sessionID string, messages []Message, requestModel, actualModel string) {
	w.Header().Set("Content-Type", "application/json")

	resp := makeHeckRequest(question, sessionID, messages, actualModel)
	scanner := bufio.NewScanner(resp.Body)

	var fullContent strings.Builder
	isAnswering := false
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			content := strings.TrimPrefix(line, "data: ")
			if content == "[ANSWER_START]" {
				isAnswering = true
				continue
			}
			if content == "[ANSWER_DONE]" {
				isAnswering = false
				break
			}
			if isAnswering {
				fullContent.WriteString(content)
			}
		}
	}

	response := OpenAIResponse{
		ID:      sessionID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   requestModel,
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: fullContent.String(),
				},
				FinishReason: "stop",
			},
		},
	}

	json.NewEncoder(w).Encode(response)
}

func makeHeckRequest(question, sessionID string, messages []Message, actualModel string) *http.Response {
	url := "https://gateway.aiapilab.com/api/ha/v1/chat"

	var previousQuestion, previousAnswer string
	messagesLen := len(messages)
	if messagesLen >= 2 {
		for i := messagesLen - 2; i >= 0; i-- {
			if messages[i].Role == "user" {
				previousQuestion = messages[i].Content
				if i+1 < messagesLen && messages[i+1].Role == "assistant" {
					previousAnswer = messages[i+1].Content
				}
				break
			}
		}
	}

	requestBody := map[string]interface{}{
		"model":            actualModel,
		"question":         question,
		"language":         "Chinese",
		"sessionId":        sessionID,
		"previousQuestion": previousQuestion,
		"previousAnswer":   previousAnswer,
	}

	jsonData, _ := json.Marshal(requestBody)
	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
	req.Header.Set("host", "gateway.aiapilab.com")

	client := &http.Client{}
	resp, _ := client.Do(req)
	return resp
}
