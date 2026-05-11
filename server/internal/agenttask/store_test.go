package agenttask

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
)

func TestStoreTaskEventAndArtifactLifecycle(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store, err := OpenStore(filepath.Join(root, "tasks.db"), filepath.Join(root, "artifacts"))
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	task, err := store.CreateTask(ctx, CreateTaskInput{
		SessionID:   "session-1",
		CharacterID: "char-1",
		Kind:        "research",
		UserRequest: "今天知乎有哪些热门信息",
	})
	if err != nil {
		t.Fatalf("CreateTask: %v", err)
	}
	if task.Status != StatusQueued || task.Progress != 0 {
		t.Fatalf("unexpected task initial state: %+v", task)
	}

	event1, updated, err := store.AppendEvent(ctx, task.ID, AppendEventInput{
		EventType: "task.started",
		Status:    StatusRunning,
		Message:   "后台任务已启动。",
		Progress:  5,
	})
	if err != nil {
		t.Fatalf("AppendEvent started: %v", err)
	}
	if event1.Seq != 1 || updated.Status != StatusRunning || updated.Progress != 5 {
		t.Fatalf("unexpected started event/task: event=%+v task=%+v", event1, updated)
	}

	artifact, err := store.CreateArtifact(ctx, task.ID, CreateArtifactInput{
		Title:   "知乎热点资料",
		Content: "# 知乎热点\n",
	})
	if err != nil {
		t.Fatalf("CreateArtifact: %v", err)
	}

	event2, updated, err := store.AppendEvent(ctx, task.ID, AppendEventInput{
		EventType: "task.completed",
		Status:    StatusCompleted,
		Message:   "任务完成。",
		Progress:  100,
	})
	if err != nil {
		t.Fatalf("AppendEvent completed: %v", err)
	}
	if event2.Seq != 2 || updated.Status != StatusCompleted || updated.FinishedAt == nil {
		t.Fatalf("unexpected completed event/task: event=%+v task=%+v", event2, updated)
	}

	events, err := store.ListEventsAfter(ctx, task.ID, 0, 10)
	if err != nil {
		t.Fatalf("ListEventsAfter: %v", err)
	}
	if len(events) != 2 || events[0].Seq != 1 || events[1].Seq != 2 {
		t.Fatalf("unexpected events: %+v", events)
	}

	gotArtifact, content, err := store.GetArtifact(ctx, task.ID, artifact.ID)
	if err != nil {
		t.Fatalf("GetArtifact: %v", err)
	}
	if gotArtifact.Title != "知乎热点资料" || string(content) != "# 知乎热点\n" {
		t.Fatalf("unexpected artifact: artifact=%+v content=%q", gotArtifact, content)
	}
}

func TestStoreRejectsLateEventAndArtifactAfterTerminal(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store, err := OpenStore(filepath.Join(root, "tasks.db"), filepath.Join(root, "artifacts"))
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	task, err := store.CreateTask(ctx, CreateTaskInput{
		SessionID:   "session-1",
		UserRequest: "取消后不要再写入",
	})
	if err != nil {
		t.Fatalf("CreateTask: %v", err)
	}
	if _, _, err := store.AppendEvent(ctx, task.ID, AppendEventInput{
		EventType: "task.cancelled",
		Status:    StatusCancelled,
		Message:   "任务已取消。",
		Progress:  0,
	}); err != nil {
		t.Fatalf("cancel task: %v", err)
	}

	if _, _, err := store.AppendEvent(ctx, task.ID, AppendEventInput{
		EventType: "task.completed",
		Status:    StatusCompleted,
		Message:   "迟到完成事件",
		Progress:  100,
	}); !errors.Is(err, ErrTerminal) {
		t.Fatalf("expected ErrTerminal for late event, got %v", err)
	}
	if _, err := store.CreateArtifact(ctx, task.ID, CreateArtifactInput{
		Title:   "迟到 artifact",
		Content: "# should not exist\n",
	}); !errors.Is(err, ErrTerminal) {
		t.Fatalf("expected ErrTerminal for late artifact, got %v", err)
	}

	events, err := store.ListEventsAfter(ctx, task.ID, 0, 10)
	if err != nil {
		t.Fatalf("ListEventsAfter: %v", err)
	}
	if len(events) != 1 || events[0].Status != StatusCancelled {
		t.Fatalf("unexpected events after terminal protection: %+v", events)
	}
}

func TestStoreLatestActiveTask(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store, err := OpenStore(filepath.Join(root, "tasks.db"), filepath.Join(root, "artifacts"))
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	first, err := store.CreateTask(ctx, CreateTaskInput{SessionID: "session-1", UserRequest: "第一个任务"})
	if err != nil {
		t.Fatalf("CreateTask first: %v", err)
	}
	if _, _, err := store.AppendEvent(ctx, first.ID, AppendEventInput{EventType: "task.completed", Status: StatusCompleted, Progress: 100}); err != nil {
		t.Fatalf("complete first: %v", err)
	}
	second, err := store.CreateTask(ctx, CreateTaskInput{SessionID: "session-1", UserRequest: "第二个任务"})
	if err != nil {
		t.Fatalf("CreateTask second: %v", err)
	}
	if _, _, err := store.AppendEvent(ctx, second.ID, AppendEventInput{EventType: "task.started", Status: StatusRunning, Progress: 10}); err != nil {
		t.Fatalf("start second: %v", err)
	}

	active, err := store.LatestActiveTask(ctx, "session-1")
	if err != nil {
		t.Fatalf("LatestActiveTask: %v", err)
	}
	if active.ID != second.ID {
		t.Fatalf("expected second active task, got %+v", active)
	}
}
