package agenttask

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/cyberverse/server/internal/ws"
)

type Config struct {
	Enabled                  bool
	WorkerURL                string
	InternalToken            string
	MaxActiveTasksPerSession int
}

type Service struct {
	store       *Store
	hub         *ws.Hub
	worker      *WorkerClient
	enabled     bool
	maxActive   int
	statusTimer time.Duration
	onEvent     func(*Task, *Event)
}

func NewService(store *Store, hub *ws.Hub, cfg Config) *Service {
	maxActive := cfg.MaxActiveTasksPerSession
	if maxActive <= 0 {
		maxActive = 3
	}
	return &Service{
		store:       store,
		hub:         hub,
		worker:      NewWorkerClient(cfg.WorkerURL, cfg.InternalToken),
		enabled:     cfg.Enabled,
		maxActive:   maxActive,
		statusTimer: 5 * time.Second,
	}
}

func (s *Service) Enabled() bool {
	return s != nil && s.enabled && s.store != nil
}

func (s *Service) Store() *Store {
	if s == nil {
		return nil
	}
	return s.store
}

func (s *Service) SetEventHandler(handler func(*Task, *Event)) {
	if s == nil {
		return
	}
	s.onEvent = handler
}

func (s *Service) CreateTask(ctx context.Context, in CreateTaskInput) (*Task, error) {
	if !s.Enabled() {
		return nil, errors.New("agent task service is disabled")
	}
	count, err := s.store.ActiveTaskCount(ctx, in.SessionID)
	if err != nil {
		return nil, err
	}
	if count >= s.maxActive {
		return nil, fmt.Errorf("session already has %d active tasks", count)
	}
	task, err := s.store.CreateTask(ctx, in)
	if err != nil {
		return nil, err
	}
	if _, _, err := s.AppendEvent(ctx, task.ID, AppendEventInput{
		EventType: "task.queued",
		Status:    StatusQueued,
		Message:   "任务已加入队列。",
		Progress:  0,
	}); err != nil {
		return nil, err
	}
	go s.dispatchTask(task.ID)
	return task, nil
}

func (s *Service) dispatchTask(taskID string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	task, err := s.store.GetTask(ctx, taskID)
	if err != nil {
		log.Printf("agent task: load task before dispatch failed task=%s: %v", taskID, err)
		return
	}
	if task.Status.IsTerminal() {
		return
	}
	if s.worker == nil || !s.worker.Configured() {
		_, _, _ = s.AppendEvent(context.Background(), taskID, AppendEventInput{
			EventType: "task.failed",
			Status:    StatusFailed,
			Message:   "Agent worker 未配置，任务无法启动。",
			Progress:  task.Progress,
		})
		return
	}
	if _, _, err := s.AppendEvent(context.Background(), taskID, AppendEventInput{
		EventType: "task.started",
		Status:    StatusRunning,
		Message:   "后台任务已启动。",
		Progress:  5,
	}); err != nil {
		if errors.Is(err, ErrTerminal) {
			return
		}
		log.Printf("agent task: append started event failed task=%s: %v", taskID, err)
	}
	task, err = s.store.GetTask(context.Background(), taskID)
	if err != nil {
		log.Printf("agent task: reload task before worker dispatch failed task=%s: %v", taskID, err)
		return
	}
	if task.Status.IsTerminal() {
		return
	}
	if err := s.worker.RunTask(context.Background(), task); err != nil {
		log.Printf("agent task: worker dispatch failed task=%s: %v", taskID, err)
		_, _, _ = s.AppendEvent(context.Background(), taskID, AppendEventInput{
			EventType: "task.failed",
			Status:    StatusFailed,
			Message:   "启动后台 Agent Worker 失败：" + err.Error(),
			Progress:  task.Progress,
		})
	}
}

func (s *Service) AppendEvent(ctx context.Context, taskID string, in AppendEventInput) (*Event, *Task, error) {
	if s == nil || s.store == nil {
		return nil, nil, errors.New("task service is not configured")
	}
	event, task, err := s.store.AppendEvent(ctx, taskID, in)
	if err != nil {
		return nil, nil, err
	}
	s.broadcastTaskEvent(task, event)
	if s.onEvent != nil {
		go s.onEvent(task, event)
	}
	return event, task, nil
}

func (s *Service) CreateArtifact(ctx context.Context, taskID string, in CreateArtifactInput) (*Artifact, error) {
	if s == nil || s.store == nil {
		return nil, errors.New("task service is not configured")
	}
	artifact, err := s.store.CreateArtifact(ctx, taskID, in)
	if err != nil {
		return nil, err
	}
	payload, _ := json.Marshal(map[string]any{
		"artifact_id": artifact.ID,
		"title":       artifact.Title,
		"type":        artifact.Type,
	})
	_, _, _ = s.AppendEvent(ctx, taskID, AppendEventInput{
		EventType: "artifact.created",
		Status:    StatusRunning,
		Message:   "已生成一份资料：" + artifact.Title,
		Progress:  90,
		Payload:   payload,
	})
	return artifact, nil
}

func (s *Service) CancelTask(ctx context.Context, taskID string) (*Task, error) {
	if s == nil || s.store == nil {
		return nil, errors.New("task service is not configured")
	}
	task, err := s.store.GetTask(ctx, taskID)
	if err != nil {
		return nil, err
	}
	if task.Status.IsTerminal() {
		return task, nil
	}
	if s.worker != nil && s.worker.Configured() {
		go func() {
			if err := s.worker.CancelTask(context.Background(), taskID); err != nil {
				log.Printf("agent task: worker cancel failed task=%s: %v", taskID, err)
			}
		}()
	}
	_, updated, err := s.AppendEvent(ctx, taskID, AppendEventInput{
		EventType: "task.cancelled",
		Status:    StatusCancelled,
		Message:   "任务已取消。",
		Progress:  task.Progress,
	})
	return updated, err
}

func (s *Service) LatestActiveTask(ctx context.Context, sessionID string) (*Task, error) {
	if s == nil || s.store == nil {
		return nil, errors.New("task service is not configured")
	}
	return s.store.LatestActiveTask(ctx, sessionID)
}

func (s *Service) RecentEventsSummary(ctx context.Context, taskID string, afterSeq int64, limit int) ([]Event, error) {
	if s == nil || s.store == nil {
		return nil, errors.New("task service is not configured")
	}
	return s.store.ListEventsAfter(ctx, taskID, afterSeq, limit)
}

func (s *Service) broadcastTaskEvent(task *Task, event *Event) {
	if s == nil || s.hub == nil || task == nil || event == nil {
		return
	}
	payload := map[string]any{
		"type":       "task_event",
		"task_id":    task.ID,
		"session_id": task.SessionID,
		"seq":        event.Seq,
		"event_type": event.EventType,
		"status":     event.Status,
		"message":    event.Message,
		"progress":   event.Progress,
		"created_at": event.CreatedAt,
		"task":       task,
	}
	if len(event.Payload) > 0 {
		payload["payload"] = event.Payload
	}
	s.hub.BroadcastJSON(task.SessionID, payload)
}

type WorkerClient struct {
	baseURL       string
	internalToken string
	client        *http.Client
}

func NewWorkerClient(baseURL, internalToken string) *WorkerClient {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	return &WorkerClient{
		baseURL:       baseURL,
		internalToken: strings.TrimSpace(internalToken),
		client:        &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *WorkerClient) Configured() bool {
	return c != nil && c.baseURL != ""
}

func (c *WorkerClient) RunTask(ctx context.Context, task *Task) error {
	if task == nil {
		return errors.New("task is nil")
	}
	return c.postJSON(ctx, fmt.Sprintf("%s/v1/tasks/%s/run", c.baseURL, task.ID), task)
}

func (c *WorkerClient) CancelTask(ctx context.Context, taskID string) error {
	return c.postJSON(ctx, fmt.Sprintf("%s/v1/tasks/%s/cancel", c.baseURL, taskID), map[string]string{"task_id": taskID})
}

func (c *WorkerClient) postJSON(ctx context.Context, url string, body any) error {
	if !c.Configured() {
		return errors.New("worker url is not configured")
	}
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.internalToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.internalToken)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("worker returned %s", resp.Status)
	}
	return nil
}
