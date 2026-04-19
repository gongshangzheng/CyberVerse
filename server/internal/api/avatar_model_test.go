package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/config"
	"github.com/cyberverse/server/internal/inference"
	"github.com/cyberverse/server/internal/orchestrator"
	pb "github.com/cyberverse/server/internal/pb"
)

type fakeInferenceService struct {
	avatarInfo *pb.AvatarInfo
	infoErr    error
}

func (f *fakeInferenceService) HealthCheck(ctx context.Context) error {
	_, err := f.AvatarInfo(ctx)
	return err
}

func (f *fakeInferenceService) AvatarInfo(context.Context) (*pb.AvatarInfo, error) {
	if f.infoErr != nil {
		return nil, f.infoErr
	}
	return f.avatarInfo, nil
}

func (f *fakeInferenceService) SetAvatar(context.Context, string, []byte, string) error { return nil }
func (f *fakeInferenceService) GenerateAvatarStream(context.Context, <-chan *pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}
func (f *fakeInferenceService) GenerateAvatar(context.Context, []*pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}
func (f *fakeInferenceService) GenerateLLMStream(context.Context, string, []inference.ChatMessage, inference.LLMConfig) (<-chan *pb.LLMChunk, <-chan error) {
	ch := make(chan *pb.LLMChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *fakeInferenceService) SynthesizeSpeechStream(context.Context, <-chan string) (<-chan *pb.AudioChunk, <-chan error) {
	ch := make(chan *pb.AudioChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *fakeInferenceService) TranscribeStream(context.Context, <-chan []byte) (<-chan *pb.TranscriptEvent, <-chan error) {
	ch := make(chan *pb.TranscriptEvent)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *fakeInferenceService) ConverseStream(context.Context, <-chan []byte, inference.VoiceLLMSessionConfig) (<-chan *pb.VoiceLLMOutput, <-chan error) {
	ch := make(chan *pb.VoiceLLMOutput)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *fakeInferenceService) Interrupt(context.Context, string) error { return nil }
func (f *fakeInferenceService) Close() error                            { return nil }

func newAvatarModelTestRouter(t *testing.T, activeModel string) (*Router, *character.Store) {
	t.Helper()

	root := t.TempDir()
	configPath := filepath.Join(root, "cyberverse_config.yaml")
	configYAML := `server:
  host: "0.0.0.0"
  http_port: 8080
  grpc_port: 50051
inference:
  avatar:
    default: "flash_head"
    runtime:
      cuda_visible_devices: "0,1"
      world_size: 2
    flash_head:
      plugin_class: "inference.plugins.avatar.flash_head_plugin.FlashHeadAvatarPlugin"
      checkpoint_dir: "/tmp/flash"
      wav2vec_dir: "/tmp/wav2vec"
      model_type: "pro"
    live_act:
      plugin_class: "inference.plugins.avatar.live_act_plugin.LiveActAvatarPlugin"
      ckpt_dir: "/tmp/live_act"
      wav2vec_dir: "/tmp/live_wav2vec"
      fps: 24
`
	if err := os.WriteFile(configPath, []byte(configYAML), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "models", "flash_head", "configs"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(root, "models", "flash_head", "configs", "infer_params.yaml"),
		[]byte("tgt_fps: 25\nframe_num: 33\n"),
		0644,
	); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "models", "live_act"), 0755); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.Load(configPath)
	if err != nil {
		t.Fatal(err)
	}
	charStore, err := character.NewStore(filepath.Join(root, "characters"))
	if err != nil {
		t.Fatal(err)
	}
	inf := &fakeInferenceService{
		avatarInfo: &pb.AvatarInfo{ModelName: "avatar." + activeModel, OutputFps: 24},
	}
	orch := orchestrator.New(inf, nil, orchestrator.NewSessionManager(4), nil, charStore)
	return NewRouter(orchestrator.NewSessionManager(4), orch, nil, nil, cfg, charStore, "", configPath), charStore
}

func TestGetAvatarModelInfoUsesRuntimeModel(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	req := httptest.NewRequest("GET", "/api/v1/config/avatar-model", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp avatarModelInfoResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.ActiveModel != "live_act" {
		t.Fatalf("expected active_model live_act, got %q", resp.ActiveModel)
	}
	if resp.ConfiguredDefaultModel != "flash_head" {
		t.Fatalf("expected configured_default_model flash_head, got %q", resp.ConfiguredDefaultModel)
	}
	for _, model := range resp.Models {
		if model.Name == "runtime" {
			t.Fatalf("did not expect runtime helper node to appear as an avatar model")
		}
	}
	if resp.ConfigStatus.HasInferParams {
		t.Fatalf("expected live_act infer params to be absent")
	}
}

func TestGetLaunchConfigUsesSharedAvatarRuntimeGPUSettingsAndSkipsMissingInferParams(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	req := httptest.NewRequest("GET", "/api/v1/config/launch?model=flash_head", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp launchConfigResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp.ActiveModel != "live_act" {
		t.Fatalf("expected active_model live_act, got %q", resp.ActiveModel)
	}
	for _, section := range resp.Sections {
		if section.Title == "视频输出" {
			t.Fatalf("did not expect 视频输出 section when infer_params.yaml is missing")
		}
		if section.Title != "GPU 配置" {
			continue
		}
		paths := map[string]bool{}
		for _, param := range section.Params {
			paths[param.Path] = true
		}
		if !paths["inference.avatar.runtime.cuda_visible_devices"] {
			t.Fatalf("expected shared avatar runtime cuda_visible_devices in GPU section")
		}
		if !paths["inference.avatar.runtime.world_size"] {
			t.Fatalf("expected shared avatar runtime world_size in GPU section")
		}
		if paths["inference.avatar.live_act.world_size"] {
			t.Fatalf("did not expect live_act-specific world_size once shared runtime is present")
		}
	}
}

func TestUpdateLaunchConfigRejectsNonActiveModel(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	body := `{"model":"flash_head","params":[{"path":"inference.avatar.flash_head.world_size","value":1}]}`
	req := httptest.NewRequest("PUT", "/api/v1/config/launch", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestUpdateLaunchConfigAllowsSharedAvatarRuntimeUpdates(t *testing.T) {
	r, _ := newAvatarModelTestRouter(t, "live_act")

	body := `{"model":"live_act","params":[{"path":"inference.avatar.runtime.world_size","value":1}]}`
	req := httptest.NewRequest("PUT", "/api/v1/config/launch", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	doc, err := config.ReadYAMLNode(r.configPath)
	if err != nil {
		t.Fatal(err)
	}
	node, err := config.GetNodeAtPath(doc, "inference.avatar.runtime.world_size")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(config.NodeValue(node, true)); got != "1" {
		t.Fatalf("expected shared world_size to be updated to 1, got %#v", got)
	}
}

func TestCreateSessionWithCharacterUsesActiveRuntimeModelOnly(t *testing.T) {
	r, charStore := newAvatarModelTestRouter(t, "live_act")
	char, err := charStore.Create(&character.Character{
		Name:      "Character Session",
		VoiceType: "温柔文雅",
	})
	if err != nil {
		t.Fatal(err)
	}

	body := `{"mode":"voice_llm","character_id":"` + char.ID + `"}`
	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d", w.Code)
	}
}

func TestCreateSessionRejectsWhenActiveRuntimeModelUnavailable(t *testing.T) {
	charStore, err := character.NewStore(filepath.Join(t.TempDir(), "characters"))
	if err != nil {
		t.Fatal(err)
	}
	char, err := charStore.Create(&character.Character{
		Name:      "Unavailable",
		VoiceType: "温柔文雅",
	})
	if err != nil {
		t.Fatal(err)
	}

	mgr := orchestrator.NewSessionManager(4)
	orch := orchestrator.New(&fakeInferenceService{
		infoErr: errors.New("inference unavailable"),
	}, nil, mgr, nil, charStore)
	r := NewRouter(mgr, orch, nil, nil, nil, charStore, "", "")

	body := `{"mode":"voice_llm","character_id":"` + char.ID + `"}`
	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}
