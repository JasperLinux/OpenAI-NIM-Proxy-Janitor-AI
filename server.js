const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "10mb" }));

// --- CONFIG ---
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// Change these to true if you want reasoning/thinking features
const SHOW_REASONING = false;
const ENABLE_THINKING = true;

// When Janitor AI sends "gpt-4o", your proxy swaps it for the NVIDIA model on the right.
// Update these to whatever models are currently available on build.nvidia.com
const MODEL_MAP = {
  "gpt-3.5-turbo": "z-ai/glm5",
  "gpt-4": "moonshotai/kimi-k2.5",
  "gpt-4-turbo": "qwen/qwen3.5-397b-a17b",
  "gpt-4o": "z-ai/glm5",
  "claude-3-opus": "moonshotai/kimi-k2.5",
  "claude-3-sonnet": "qwen/qwen3.5-397b-a17b",
  "gemini-pro": "z-ai/glm5",
};

// Fallback: if the model name isn't in the map, pick one based on keywords
function resolveFallback(name) {
  const lower = name.toLowerCase();
  if (
    lower.includes("gpt-4") ||
    lower.includes("opus") ||
    lower.includes("405b")
  )
    return "meta/llama-3.1-405b-instruct";
  if (
    lower.includes("claude") ||
    lower.includes("gemini") ||
    lower.includes("70b")
  )
    return "meta/llama-3.1-70b-instruct";
  return "meta/llama-3.1-8b-instruct";
}

// --- ROUTES ---

// Health check — hit this in your browser to confirm the proxy is alive
app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "nim-proxy" });
});

// OpenAI-compatible model list
app.get("/v1/models", (_req, res) => {
  const data = Object.keys(MODEL_MAP).map((id) => ({
    id,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "nim-proxy",
  }));
  res.json({ object: "list", data });
});

// Main endpoint — this is what Janitor AI actually calls
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const {
      model = "gpt-4o",
      messages,
      temperature,
      max_tokens,
      stream,
    } = req.body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: "messages array is required",
          type: "invalid_request_error",
        },
      });
    }

    const nimModel = MODEL_MAP[model] || resolveFallback(model);

    // Build the request body for NVIDIA
    const nimBody = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens || 4096,
      stream: !!stream,
    };

    // Only add thinking param if enabled (some models support this)
    if (ENABLE_THINKING) {
      nimBody.chat_template_kwargs = { thinking: true };
    }

    const nimResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimBody,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json",
        },
        responseType: stream ? "stream" : "json",
        timeout: 120000, // 2 min timeout
      }
    );

    // --- STREAMING ---
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      let buffer = "";
      let inReasoning = false;

      nimResponse.data.on("data", (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          if (line.includes("[DONE]")) {
            res.write("data: [DONE]\n\n");
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (!delta) {
              res.write(`data: ${JSON.stringify(data)}\n\n`);
              continue;
            }

            const reasoning = delta.reasoning_content;
            const content = delta.content;

            // Build the output content
            let output = "";

            if (SHOW_REASONING && reasoning) {
              if (!inReasoning) {
                output += "<think>\n";
                inReasoning = true;
              }
              output += reasoning;
            }

            if (content) {
              if (inReasoning) {
                output += "\n</think>\n\n";
                inReasoning = false;
              }
              output += content;
            }

            // Only send chunks that have actual content
            if (output) {
              delta.content = output;
            } else if (!content && !reasoning) {
              // Keep empty deltas for finish_reason signals etc.
            } else {
              delta.content = "";
            }
            delete delta.reasoning_content;

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            // Skip unparseable lines
          }
        }
      });

      nimResponse.data.on("end", () => res.end());
      nimResponse.data.on("error", () => res.end());
      return;
    }

    // --- NON-STREAMING ---
    const choices = (nimResponse.data.choices || []).map((choice) => {
      let content = choice.message?.content || "";

      if (SHOW_REASONING && choice.message?.reasoning_content) {
        content =
          "<think>\n" +
          choice.message.reasoning_content +
          "\n</think>\n\n" +
          content;
      }

      return {
        index: choice.index,
        message: { role: "assistant", content },
        finish_reason: choice.finish_reason,
      };
    });

    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices,
      usage: nimResponse.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      },
    });
  } catch (error) {
    const status = error.response?.status || 500;
    const msg =
      error.response?.data?.error?.message ||
      error.message ||
      "Internal server error";
    console.error(`Proxy error [${status}]: ${msg}`);
    res.status(status).json({
      error: { message: msg, type: "proxy_error", code: status },
    });
  }
});

// Everything else → 404
app.use((_req, res) => {
  res.status(404).json({
    error: { message: "Not found", type: "invalid_request_error", code: 404 },
  });
});

app.listen(PORT, () => {
  console.log(`NIM proxy running on port ${PORT}`);
});
