# import os, json, time, threading, queue, asyncio, re
# from pathlib import Path
# from typing import Optional, Dict, Any, List

# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse, JSONResponse
# import uvicorn

# # Load .env early
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except Exception:
#     pass

# # Google Cloud Speech-to-Text
# from google.cloud import speech
# from google.oauth2 import service_account

# # Supabase
# from supabase import create_client, Client

# # Ollama (Cloud/local)
# from ollama import Client as OllamaClient

# # ─────────────────────────────────────────────────────────────
# # FastAPI app + CORS + static (docs disabled)
# # ─────────────────────────────────────────────────────────────
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# HERE = Path(__file__).resolve().parent
# CANDIDATES = [HERE.parent / "static", HERE / "static"]
# for d in CANDIDATES:
#     if d.exists():
#         STATIC_DIR = d
#         break
# else:
#     STATIC_DIR = CANDIDATES[0]
# print(f"[startup] static dir -> {STATIC_DIR} (exists={STATIC_DIR.exists()})")
# if STATIC_DIR.exists():
#     app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# @app.get("/health")
# def health():
#     return JSONResponse({"ok": True})

# @app.get("/")
# def index():
#     idx = STATIC_DIR / "index.html"
#     if idx.exists():
#         return HTMLResponse(idx.read_text(encoding="utf-8"))
#     return HTMLResponse("<h2>UI missing</h2>")

# # ─────────────────────────────────────────────────────────────
# # Supabase (.env)
# # ─────────────────────────────────────────────────────────────
# SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
# SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
# _sb: Optional[Client] = None

# def supabase() -> Client:
#     global _sb
#     if _sb is None:
#         if not SUPABASE_ENABLED:
#             raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
#         _sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
#     return _sb

# # ─────────────────────────────────────────────────────────────
# # Ollama (.env)
# # ─────────────────────────────────────────────────────────────
# OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
# OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# def get_llm() -> Optional[OllamaClient]:
#     if not OLLAMA_HOST or not OLLAMA_MODEL:
#         return None
#     headers = None
#     if OLLAMA_HOST.startswith("https://ollama.com"):
#         if not OLLAMA_API_KEY:
#             return None
#         headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
#     try:
#         return OllamaClient(host=OLLAMA_HOST, headers=headers)
#     except Exception:
#         return None

# def _ollama_chat_json(prompt: str, temperature: float = 0.2) -> dict:
#     client = get_llm()
#     if not client:
#         raise HTTPException(501, "Ollama Cloud not configured on server")
#     messages = [
#         {"role": "system", "content": "You are an expert technical interview copilot. Always return STRICT JSON."},
#         {"role": "user", "content": prompt},
#     ]
#     try:
#         resp = client.chat(model=OLLAMA_MODEL, messages=messages, stream=False, options={"temperature": temperature})
#     except Exception as e:
#         raise HTTPException(500, f"Ollama error: {e}")
#     content = (resp.get("message") or {}).get("content", "").strip()
#     try:
#         return json.loads(content)
#     except Exception:
#         return {"raw": content}

# # ─────────────────────────────────────────────────────────────
# # Google STT helpers
# # ─────────────────────────────────────────────────────────────
# MAX_STREAM_SECONDS = 290

# def get_speech_client() -> speech.SpeechClient:
#     env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
#     if env_path and os.path.exists(env_path):
#         creds = service_account.Credentials.from_service_account_file(env_path)
#         print(f"[startup] using credentials from env: {env_path}")
#         return speech.SpeechClient(credentials=creds)
#     local = Path(__file__).with_name("service-account.json")
#     if local.exists():
#         creds = service_account.Credentials.from_service_account_file(str(local))
#         print("[startup] using credentials from backend/service-account.json")
#         return speech.SpeechClient(credentials=creds)
#     print("[startup] using ADC")
#     return speech.SpeechClient()

# def make_recognition_config():
#     return speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=16000,
#         language_code="en-US",
#         enable_automatic_punctuation=True,
#         use_enhanced=True,
#         model="default",
#     )

# def make_streaming_config():
#     return speech.StreamingRecognitionConfig(
#         config=make_recognition_config(),
#         interim_results=True,
#         single_utterance=False,
#     )

# # ─────────────────────────────────────────────────────────────
# # Session
# # ─────────────────────────────────────────────────────────────
# @app.post("/session/start")
# async def start_session(payload: Dict[str, str]):
#     meeting_id = payload.get("meeting_id")
#     if not meeting_id:
#         raise HTTPException(400, "meeting_id required")
#     if not SUPABASE_ENABLED:
#         raise HTTPException(500, "Supabase not configured on server (.env missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)")

#     row = {"meeting_id": meeting_id}
#     for k in ("jd", "resume", "interviewer_name", "candidate_name"):
#         v = payload.get(k)
#         if v is not None:
#             row[k] = v

#     # Do NOT touch transcript here; append happens via STT
#     supabase().table("meetings").upsert(row, on_conflict="meeting_id").execute()
#     return {"session_id": meeting_id}

# @app.get("/session/{session_id}/turns")
# async def get_turns(session_id: str):
#     mtg = _sb_get_meeting(session_id)
#     pseudo = _parse_transcript_to_turns(
#         mtg.get("transcript") or "",
#         mtg.get("interviewer_name") or "",
#         mtg.get("candidate_name") or "",
#     )
#     return {"session_id": session_id, "turns": pseudo}

# # ─────────────────────────────────────────────────────────────
# # WebSocket STT
# # ─────────────────────────────────────────────────────────────
# @app.websocket("/ws/{source}")
# async def websocket_stt(websocket: WebSocket, source: str):
#     if source not in ("mic", "tab"):
#         await websocket.close(code=1008)
#         return
#     await websocket.accept()
#     qp = websocket.query_params
#     session_id = qp.get("session_id")
#     speaker = qp.get("speaker") or ("interviewer" if source == "mic" else "candidate")
#     speaker_name = qp.get("speaker_name") or speaker

#     print(f"[ws:{source}] connected (session={session_id}, speaker={speaker}, name={speaker_name})")

#     try:
#         stt_client = get_speech_client()
#     except Exception as e:
#         print(f"[stt-init] error: {e}")
#         await websocket.close()
#         return

#     audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
#     out_q: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()

#     worker = threading.Thread(
#         target=google_stt_worker,
#         args=(source, session_id, speaker, speaker_name, stt_client, audio_q, asyncio.get_running_loop(), out_q),
#         daemon=True,
#     )
#     worker.start()

#     sender_task = asyncio.create_task(result_sender(websocket, out_q, source))

#     try:
#         while True:
#             try:
#                 msg = await websocket.receive()
#             except WebSocketDisconnect:
#                 print(f"[ws:{source}] disconnected")
#                 break
#             except RuntimeError as e:
#                 if "disconnect message has been received" in str(e).lower():
#                     print(f"[ws:{source}] disconnect acknowledged")
#                     break
#                 raise
#             if msg.get("bytes") is not None:
#                 audio_q.put(msg["bytes"])  # PCM16 bytes
#     finally:
#         audio_q.put(None)
#         try:
#             await asyncio.wait_for(sender_task, timeout=2)
#         except asyncio.TimeoutError:
#             sender_task.cancel()
#         print(f"[ws:{source}] done")

# def google_stt_worker(
#     source: str,
#     session_id: Optional[str],
#     speaker: str,
#     speaker_name: str,
#     client: speech.SpeechClient,
#     audio_q: "queue.Queue[Optional[bytes]]",
#     loop: asyncio.AbstractEventLoop,
#     out_q: "asyncio.Queue[Optional[dict]]",
# ):
#     def gen_with_config(stop_flag: Dict[str, bool]):
#         yield speech.StreamingRecognizeRequest(streaming_config=make_streaming_config())
#         while True:
#             chunk = audio_q.get()
#             if chunk is None:
#                 stop_flag["stop"] = True
#                 break
#             yield speech.StreamingRecognizeRequest(audio_content=chunk)

#     def audio_only(stop_flag: Dict[str, bool]):
#         while True:
#             chunk = audio_q.get()
#             if chunk is None:
#                 stop_flag["stop"] = True
#                 break
#             yield speech.StreamingRecognizeRequest(audio_content=chunk)

#     while True:
#         stop_flag = {"stop": False}
#         start = time.time()
#         try:
#             try:
#                 responses = client.streaming_recognize(requests=gen_with_config(stop_flag))
#             except TypeError:
#                 responses = client.streaming_recognize(make_streaming_config(), audio_only(stop_flag))

#             for resp in responses:
#                 if not resp.results:
#                     if time.time() - start > MAX_STREAM_SECONDS:
#                         break
#                     continue
#                 for result in resp.results:
#                     text = result.alternatives[0].transcript
#                     is_final = result.is_final

#                     asyncio.run_coroutine_threadsafe(
#                         out_q.put(
#                             {
#                                 "source": source,
#                                 "speaker": speaker,
#                                 "speaker_name": speaker_name,
#                                 "text": text,
#                                 "final": is_final,
#                             }
#                         ),
#                         loop,
#                     )

#                     # Append only to meetings.transcript (no 'turns' table)
#                     if SUPABASE_ENABLED and session_id and is_final and text.strip():
#                         try:
#                             _append_transcript_line(session_id, speaker_name, text.strip())
#                         except Exception as e:
#                             print(f"[stt:{source}] transcript append error: {e}")

#                 if time.time() - start > MAX_STREAM_SECONDS:
#                     print("[stt] rotating stream to avoid 305s limit…")
#                     break

#         except Exception as e:
#             if "Audio Timeout Error" in str(e):
#                 print("[stt] timeout; restarting stream…")
#                 continue
#             print(f"[stt:{source}] error: {e}")
#             continue
#         finally:
#             if stop_flag["stop"]:
#                 break

#     asyncio.run_coroutine_threadsafe(out_q.put(None), loop)

# async def result_sender(websocket: WebSocket, out_q: "asyncio.Queue[Optional[dict]]", source: str):
#     while True:
#         item = await out_q.get()
#         if item is None:
#             break
#         try:
#             await websocket.send_text(json.dumps(item))
#         except Exception as e:
#             print(f"[ws:{source}] send error: {e}")
#             break

# # ─────────────────────────────────────────────────────────────
# # DB helpers (meetings + transcript + ai_report)
# # ─────────────────────────────────────────────────────────────
# def _sb_get_meeting(session_id: str) -> dict:
#     if not SUPABASE_ENABLED:
#         return {}
#     try:
#         m = (
#             supabase()
#             .table("meetings")
#             .select("*")
#             .eq("meeting_id", session_id)
#             .limit(1)
#             .execute()
#         )
#         if m.data:
#             return m.data[0]
#     except Exception as e:
#         print(f"[sb] get meeting error: {e}")
#     return {}

# def _sb_upd_meeting_transcript(session_id: str, new_transcript: str) -> None:
#     if not SUPABASE_ENABLED:
#         return
#     try:
#         supabase().table("meetings").update({"transcript": new_transcript}).eq("meeting_id", session_id).execute()
#     except Exception as e:
#         print(f"[sb] update transcript error: {e}")

# def _append_transcript_line(session_id: str, speaker_name: str, text: str) -> None:
#     mtg = _sb_get_meeting(session_id)
#     prev = mtg.get("transcript") or ""
#     line = f"{speaker_name}: {text}".strip()
#     new = (prev.rstrip() + "\n" + line) if prev.strip() else line
#     _sb_upd_meeting_transcript(session_id, new)

# # transcript → pseudo turns (fallback/only path now)
# def _parse_transcript_to_turns(transcript_text: str, interviewer_name: str, candidate_name: str) -> List[dict]:
#     if not (transcript_text or "").strip():
#         return []
#     out: List[dict] = []
#     lines = [ln.strip() for ln in transcript_text.splitlines() if ln.strip()]
#     for ln in lines:
#         if ":" in ln:
#             name, msg = ln.split(":", 1)
#             name = name.strip(); msg = msg.strip()
#         else:
#             if out:
#                 out[-1]["text"] += (" " + ln); continue
#             name, msg = "Unknown", ln

#         role = "unknown"
#         if interviewer_name and name.lower() == interviewer_name.lower():
#             role = "interviewer"
#         elif candidate_name and name.lower() == candidate_name.lower():
#             role = "candidate"
#         else:
#             if name.lower() in ("interviewer", "mic"):
#                 role = "interviewer"
#             elif name.lower() in ("candidate", "tab"):
#                 role = "candidate"

#         out.append({
#             "source": "mic" if role == "interviewer" else ("tab" if role == "candidate" else "unknown"),
#             "speaker": role,
#             "speaker_name": name,
#             "text": msg,
#             "is_final": True,
#         })
#     return out

# # ─────────────────────────────────────────────────────────────
# # Formatting helpers — make outputs point-wise, no asterisks
# # ─────────────────────────────────────────────────────────────
# _CODE_FENCE = re.compile(r"```.*?```", flags=re.S)

# def _strip_code_and_json(s: str) -> str:
#     if not s:
#         return ""
#     s = _CODE_FENCE.sub("", s)
#     s = s.replace("`", " ")
#     s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
#     s = re.sub(r"\*+", "", s)
#     s = re.sub(r"_+", "", s)
#     s = s.strip()
#     if s.startswith("{") and s.endswith("}"):
#         try:
#             obj = json.loads(s)
#             s = (obj.get("explanation") or obj.get("expected_answer") or "")
#         except Exception:
#             s = ""
#     return s.strip()

# def _to_bullets(s: str, max_items: int = 6, max_len: int = 180) -> str:
#     s = _strip_code_and_json(s)
#     if not s:
#         return ""
#     parts = re.split(r"[\r\n]+|(?:^|[\s])[-–•·]\s+", s)
#     parts = [p.strip(" -–•·\t\r\n") for p in parts if p and p.strip()]
#     if len(parts) <= 1:
#         parts = re.split(r"(?<=[.!?])\s+", s)
#     clean = []
#     for p in parts:
#         p = re.sub(r"\s+", " ", p).strip()
#         if not p:
#             continue
#         if len(p) > max_len:
#             p = p[:max_len-1].rstrip() + "…"
#         clean.append(p)
#         if len(clean) >= max_items:
#             break
#     if not clean:
#         return ""
#     return "\n".join(f"• {p}" for p in clean)

# # ─────────────────────────────────────────────────────────────
# # Candidate answer extraction (transcript-only)
# # ─────────────────────────────────────────────────────────────
# def _norm(s: str) -> str:
#     s = s.lower().strip()
#     s = re.sub(r"[^a-z0-9\s]+", "", s)
#     s = re.sub(r"\s+", " ", s)
#     return s

# def _is_interviewer(t: dict, interviewer_name: str) -> bool:
#     who = (t.get("speaker") or t.get("source") or "").lower()
#     name = (t.get("speaker_name") or "").lower()
#     if "interviewer" in who or "mic" in who:
#         return True
#     if interviewer_name and name and name == interviewer_name.lower():
#         return True
#     return False

# def _is_candidate(t: dict, candidate_name: str) -> bool:
#     who = (t.get("speaker") or t.get("source") or "").lower()
#     name = (t.get("speaker_name") or "").lower()
#     if "candidate" in who or "tab" in who:
#         return True
#     if candidate_name and name and name == candidate_name.lower():
#         return True
#     return False

# def _recent_turns(session_id: str, limit: int = 200) -> List[dict]:
#     mtg = _sb_get_meeting(session_id)
#     turns = _parse_transcript_to_turns(
#         mtg.get("transcript") or "",
#         mtg.get("interviewer_name") or "",
#         mtg.get("candidate_name") or "",
#     )
#     if limit and len(turns) > limit:
#         return turns[-limit:]
#     return turns

# def _turns_snippet(turns: List[dict], last_n: int = 12) -> str:
#     ctx = turns[-last_n:] if len(turns) > last_n else turns[:]
#     lines = []
#     for t in ctx:
#         who = t.get("speaker_name") or t.get("speaker") or t.get("source", "unknown")
#         text = t.get("text", "")
#         lines.append(f"{who}: {text}")
#     return "\n".join(lines)

# def _extract_candidate_answer(question_text: str, turns: List[dict], interviewer_name: str = "", candidate_name: str = "") -> str:
#     if not turns:
#         return ""
#     qn = _norm(question_text)
#     best_idx, best_score = None, 0.0
#     qtoks = set(qn.split())
#     for i, t in enumerate(turns):
#         if not _is_interviewer(t, interviewer_name):
#             continue
#         tn = _norm(t.get("text", ""))
#         if not tn:
#             continue
#         ttoks = set(tn.split())
#         if not qtoks or not ttoks:
#             continue
#         overlap = len(qtoks & ttoks) / max(1, len(qtoks))
#         if qn in tn or tn in qn:
#             overlap = 1.0
#         if overlap > best_score:
#             best_score, best_idx = overlap, i

#     if best_idx is not None:
#         out = []
#         for j in range(best_idx + 1, len(turns)):
#             t = turns[j]
#             if _is_interviewer(t, interviewer_name):
#                 break
#             if _is_candidate(t, candidate_name):
#                 out.append(t.get("text", ""))
#             if len(out) >= 6:
#                 break
#         return " ".join(x.strip() for x in out if x.strip())

#     cand = [t.get("text", "") for t in turns[-10:] if _is_candidate(t, candidate_name)]
#     return " ".join(x.strip() for x in cand if x.strip())

# # ─────────────────────────────────────────────────────────────
# # AI endpoints (kept)
# # ─────────────────────────────────────────────────────────────
# @app.post("/ai/summary")
# async def ai_summary(payload: Dict[str, str] = Body(...)):
#     session_id = (payload or {}).get("session_id")
#     if not session_id:
#         raise HTTPException(400, "session_id required")
#     meeting = _sb_get_meeting(session_id)
#     turns = _recent_turns(session_id, limit=2000)
#     resume = meeting.get("resume", "")
#     jd     = meeting.get("jd", "")
#     lines = [f"{(t.get('speaker_name') or t.get('speaker') or t.get('source'))}: {t.get('text','')}" for t in turns]
#     convo = "\n".join(lines)
#     prompt = f"""
# Return STRICT JSON with:
# candidate_summary, interview_summary, strengths (array), risks (array), recommended_next_questions (array).

# RESUME:
# {resume}

# JD:
# {jd}

# FULL TRANSCRIPT:
# {convo}
# """.strip()
#     data = _ollama_chat_json(prompt)
#     coerced = _coerce_summary_payload(data)

#     try:
#         _sb_upsert_ai_report(session_id, coerced)
#     except Exception as e:
#         print(f"[ai_summary] store report error: {e}")

#     return {"session_id": session_id, **coerced}

# @app.post("/ai/expected")
# async def ai_expected(payload: Dict[str, str] = Body(...)):
#     session_id = (payload or {}).get("session_id")
#     question   = (payload or {}).get("question", "").strip()
#     if not session_id or not question:
#         raise HTTPException(400, "session_id and question required")
#     meeting = _sb_get_meeting(session_id)
#     resume = meeting.get("resume", "")
#     jd     = meeting.get("jd", "")
#     turns  = _recent_turns(session_id, 80)
#     convo  = _turns_snippet(turns, 12)
#     prompt = f"""
# Return STRICT JSON with key: expected_answer.

# Context:
# RESUME:
# {resume}

# JD:
# {jd}

# RECENT TRANSCRIPT:
# {convo}

# QUESTION TO PREPARE EXPECTED ANSWER FOR:
# {question}

# Write a concise but complete expected answer (bullets or short paragraphs).
# """.strip()
#     data = _ollama_chat_json(prompt)
#     exp_raw = (data.get("expected_answer") or data.get("raw") or "").strip()
#     return {
#         "session_id": session_id,
#         "question": question,
#         "expected_answer": _to_bullets(exp_raw, max_items=6),
#     }

# @app.post("/ai/validate")
# async def ai_validate(payload: Dict[str, str] = Body(...)):
#     session_id = (payload or {}).get("session_id")
#     question   = (payload or {}).get("question", "").strip()
#     if not session_id or not question:
#         raise HTTPException(400, "session_id and question required")
#     meeting = _sb_get_meeting(session_id)
#     resume = meeting.get("resume", "")
#     jd     = meeting.get("jd", "")
#     turns  = _recent_turns(session_id, 300)

#     cand_answer = _extract_candidate_answer(
#         question,
#         turns,
#         interviewer_name=(meeting.get("interviewer_name") or ""),
#         candidate_name=(meeting.get("candidate_name") or ""),
#     )

#     prompt = f"""
# You must return STRICT JSON with keys:
# - verdict (one of: "RIGHT", "ALMOST", "WRONG")
# - score (0.0–1.0)
# - explanation (1–3 sentences)
# - expected_answer
# - candidate_answer

# Use:
# RESUME:
# {resume}

# JD:
# {jd}

# QUESTION:
# {question}

# CANDIDATE_ANSWER:
# {cand_answer if cand_answer else "(none)"} 

# First, infer the ideal expected_answer from JD/Resume. Then compare candidate_answer to expected_answer and fill verdict/score/explanation.
# """.strip()
#     data = _ollama_chat_json(prompt)

#     verdict = (data.get("verdict") or "").upper().strip()
#     if verdict not in {"RIGHT", "ALMOST", "WRONG"}:
#         verdict = "WRONG" if not cand_answer else "ALMOST"
#     try:
#         score = float(data.get("score", 0.0))
#     except Exception:
#         score = 0.0
#     score = max(0.0, min(1.0, score))
#     score_pct = int(round(score * 100))

#     exp_raw = (data.get("expected_answer") or "").strip()
#     explanation_raw = (data.get("explanation") or data.get("raw") or "").strip()

#     return {
#         "session_id": session_id,
#         "question": question,
#         "expected_answer": _to_bullets(exp_raw, max_items=6),
#         "candidate_answer": cand_answer,
#         "verdict": verdict,
#         "score": score,
#         "score_pct": score_pct,
#         "explanation": _to_bullets(explanation_raw, max_items=3, max_len=160),
#     }

# @app.post("/ai/questions")
# async def ai_questions(payload: Dict[str, Any] = Body(...)):
#     session_id = (payload or {}).get("session_id")
#     count = int((payload or {}).get("count", 5))
#     if not session_id:
#         raise HTTPException(400, "session_id required")
#     meeting = _sb_get_meeting(session_id)
#     resume = meeting.get("resume", "")
#     jd     = meeting.get("jd", "")
#     turns  = _recent_turns(session_id, 60)
#     convo  = _turns_snippet(turns, 10)
#     prompt = f"""
# Return STRICT JSON with key: questions (array of {count} concise follow-up questions).
# Use JD and Resume as primary guidance; if transcript context exists, adapt to it; otherwise generate from JD/Resume only.

# RESUME:
# {resume}

# JD:
# {jd}

# RECENT TRANSCRIPT:
# {convo if convo.strip() else "(none)"}
# """.strip()
#     data = _ollama_chat_json(prompt, temperature=0.6)
#     questions = data.get("questions")
#     if not isinstance(questions, list):
#         raw = (data.get("raw") or "").strip()
#         questions = [x.strip("-• ").strip() for x in raw.split("\n") if x.strip()]
#     questions = [q for q in questions if q][:count]
#     return {"session_id": session_id, "questions": questions}

# # ─────────────────────────────────────────────────────────────
# # Store AI summary into meetings.ai_report (jsonb column)
# # ─────────────────────────────────────────────────────────────
# def _sb_upsert_ai_report(session_id: str, report: dict) -> None:
#     if not SUPABASE_ENABLED:
#         return
#     try:
#         safe_report = json.loads(json.dumps(report, ensure_ascii=False))
#     except Exception as e:
#         print(f"[sb] ai_report not JSON-serializable: {e}")
#         return
#     try:
#         supabase().table("meetings").update({"ai_report": safe_report}).eq("meeting_id", session_id).execute()
#     except Exception as e:
#         print(f"[sb] update meetings.ai_report error: {e}")

# # ─────────────────────────────────────────────────────────────
# # Uvicorn entrypoint
# # ─────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     uvicorn.run("server2", host="127.0.0.1", port=8000, reload=True)

# 2nd  whole working 
import os, json, time, threading, queue, asyncio, re
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Load .env early
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Google Cloud Speech-to-Text
from google.cloud import speech
from google.oauth2 import service_account

# Supabase
from supabase import create_client, Client

# Ollama (Cloud/local)
from ollama import Client as OllamaClient

# ─────────────────────────────────────────────────────────────
# FastAPI app + CORS + static (docs offline)
# ─────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parent / "static", HERE / "static"]
for d in CANDIDATES:
    if d.exists():
        STATIC_DIR = d
        break
else:
    STATIC_DIR = CANDIDATES[0]
print(f"[startup] static dir -> {STATIC_DIR} (exists={STATIC_DIR.exists()})")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/health")
def health():
    return JSONResponse({"ok": True})

@app.get("/")
def index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>UI missing</h2>")

# Offline Swagger UI (no CDN) + cached OpenAPI
@lru_cache()
def custom_openapi():
    return get_openapi(
        title="Meet Bot API",
        version="1.0.0",
        routes=app.routes,
        description="Realtime interview STT + AI endpoints",
    )
app.openapi = custom_openapi  # type: ignore

@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    # Make sure these files exist under /static/swagger-ui/
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Meet Bot Docs",
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_ui_preset_url="/static/swagger-ui/swagger-ui-standalone-preset.js",
    )

# ─────────────────────────────────────────────────────────────
# Supabase (.env)
# ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
_sb: Optional[Client] = None

def supabase() -> Client:
    global _sb
    if _sb is None:
        if not SUPABASE_ENABLED:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        _sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _sb

# ─────────────────────────────────────────────────────────────
# Ollama (.env)
# ─────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

def get_llm() -> Optional[OllamaClient]:
    if not OLLAMA_HOST or not OLLAMA_MODEL:
        return None
    headers = None
    if OLLAMA_HOST.startswith("https://ollama.com"):
        if not OLLAMA_API_KEY:
            return None
        headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    try:
        return OllamaClient(host=OLLAMA_HOST, headers=headers)
    except Exception:
        return None

def _ollama_chat_json(prompt: str, temperature: float = 0.2) -> dict:
    client = get_llm()
    if not client:
        raise HTTPException(501, "Ollama Cloud not configured on server")
    messages = [
        {"role": "system", "content": "You are an expert technical interview copilot. Always return STRICT JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat(model=OLLAMA_MODEL, messages=messages, stream=False, options={"temperature": temperature})
    except Exception as e:
        raise HTTPException(500, f"Ollama error: {e}")
    content = (resp.get("message") or {}).get("content", "").strip()
    try:
        return json.loads(content)
    except Exception:
        return {"raw": content}

# ─────────────────────────────────────────────────────────────
# Google STT helpers
# ─────────────────────────────────────────────────────────────
MAX_STREAM_SECONDS = 290

def get_speech_client() -> speech.SpeechClient:
    # You must set GOOGLE_APPLICATION_CREDENTIALS to a valid service-account.json
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and os.path.exists(env_path):
        creds = service_account.Credentials.from_service_account_file(env_path)
        print(f"[startup] using credentials from env: {env_path}")
        return speech.SpeechClient(credentials=creds)
    local = Path(__file__).with_name("service-account.json")
    if local.exists():
        creds = service_account.Credentials.from_service_account_file(str(local))
        print("[startup] using credentials from backend/service-account.json")
        return speech.SpeechClient(credentials=creds)
    print("[startup] using ADC")
    return speech.SpeechClient()

def make_recognition_config():
    return speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
        use_enhanced=True,
        model="default",
    )

def make_streaming_config():
    return speech.StreamingRecognitionConfig(
        config=make_recognition_config(),
        interim_results=True,
        single_utterance=False,
    )

# ─────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────
@app.post("/session/start")
async def start_session(payload: Dict[str, str]):
    meeting_id = payload.get("meeting_id")
    if not meeting_id:
        raise HTTPException(400, "meeting_id required")
    if not SUPABASE_ENABLED:
        raise HTTPException(500, "Supabase not configured on server (.env missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)")

    row = {"meeting_id": meeting_id}
    for k in ("jd", "resume", "interviewer_name", "candidate_name"):
        v = payload.get(k)
        if v is not None:
            row[k] = v

    # Do NOT touch transcript here; append happens via STT
    supabase().table("meetings").upsert(row, on_conflict="meeting_id").execute()
    return {"session_id": meeting_id}

@app.get("/session/{session_id}/turns")
async def get_turns(session_id: str):
    mtg = _sb_get_meeting(session_id)
    pseudo = _parse_transcript_to_turns(
        mtg.get("transcript") or "",
        mtg.get("interviewer_name") or "",
        mtg.get("candidate_name") or "",
    )
    return {"session_id": session_id, "turns": pseudo}

# ─────────────────────────────────────────────────────────────
# WebSocket STT
# ─────────────────────────────────────────────────────────────
@app.websocket("/ws/{source}")
async def websocket_stt(websocket: WebSocket, source: str):
    if source not in ("mic", "tab"):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    qp = websocket.query_params
    session_id = qp.get("session_id")
    speaker = qp.get("speaker") or ("interviewer" if source == "mic" else "candidate")
    speaker_name = qp.get("speaker_name") or speaker

    print(f"[ws:{source}] connected (session={session_id}, speaker={speaker}, name={speaker_name})")

    try:
        stt_client = get_speech_client()
    except Exception as e:
        print(f"[stt-init] error: {e}")
        await websocket.close()
        return

    audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
    out_q: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()

    worker = threading.Thread(
        target=google_stt_worker,
        args=(source, session_id, speaker, speaker_name, stt_client, audio_q, asyncio.get_running_loop(), out_q),
        daemon=True,
    )
    worker.start()

    sender_task = asyncio.create_task(result_sender(websocket, out_q, source))

    try:
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                print(f"[ws:{source}] disconnected")
                break
            except RuntimeError as e:
                if "disconnect message has been received" in str(e).lower():
                    print(f"[ws:{source}] disconnect acknowledged")
                    break
                raise
            if msg.get("bytes") is not None:
                audio_q.put(msg["bytes"])  # PCM16 bytes
    finally:
        audio_q.put(None)
        try:
            await asyncio.wait_for(sender_task, timeout=2)
        except asyncio.TimeoutError:
            sender_task.cancel()
        print(f"[ws:{source}] done")

def google_stt_worker(
    source: str,
    session_id: Optional[str],
    speaker: str,
    speaker_name: str,
    client: speech.SpeechClient,
    audio_q: "queue.Queue[Optional[bytes]]",
    loop: asyncio.AbstractEventLoop,
    out_q: "asyncio.Queue[Optional[dict]]",
):
    def gen_with_config(stop_flag: Dict[str, bool]):
        yield speech.StreamingRecognizeRequest(streaming_config=make_streaming_config())
        while True:
            chunk = audio_q.get()
            if chunk is None:
                stop_flag["stop"] = True
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def audio_only(stop_flag: Dict[str, bool]):
        while True:
            chunk = audio_q.get()
            if chunk is None:
                stop_flag["stop"] = True
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    while True:
        stop_flag = {"stop": False}
        start = time.time()
        try:
            try:
                responses = client.streaming_recognize(requests=gen_with_config(stop_flag))
            except TypeError:
                responses = client.streaming_recognize(make_streaming_config(), audio_only(stop_flag))

            for resp in responses:
                if not resp.results:
                    if time.time() - start > MAX_STREAM_SECONDS:
                        break
                    continue
                for result in resp.results:
                    text = result.alternatives[0].transcript
                    is_final = result.is_final

                    asyncio.run_coroutine_threadsafe(
                        out_q.put(
                            {
                                "source": source,
                                "speaker": speaker,
                                "speaker_name": speaker_name,
                                "text": text,
                                "final": is_final,
                            }
                        ),
                        loop,
                    )

                    # Append only to meetings.transcript (no 'turns' table)
                    if SUPABASE_ENABLED and session_id and is_final and text.strip():
                        try:
                            _append_transcript_line(session_id, speaker_name, text.strip())
                        except Exception as e:
                            print(f"[stt:{source}] transcript append error: {e}")

                if time.time() - start > MAX_STREAM_SECONDS:
                    print("[stt] rotating stream to avoid 305s limit…")
                    break

        except Exception as e:
            if "Audio Timeout Error" in str(e):
                print("[stt] timeout; restarting stream…")
                continue
            print(f"[stt:{source}] error: {e}")
            continue
        finally:
            if stop_flag["stop"]:
                break

    asyncio.run_coroutine_threadsafe(out_q.put(None), loop)

async def result_sender(websocket: WebSocket, out_q: "asyncio.Queue[Optional[dict]]", source: str):
    while True:
        item = await out_q.get()
        if item is None:
            break
        try:
            await websocket.send_text(json.dumps(item))
        except Exception as e:
            print(f"[ws:{source}] send error: {e}")
            break

# ─────────────────────────────────────────────────────────────
# DB helpers (meetings + transcript + ai_report)
# ─────────────────────────────────────────────────────────────
def _sb_get_meeting(session_id: str) -> dict:
    if not SUPABASE_ENABLED:
        return {}
    try:
        m = (
            supabase()
            .table("meetings")
            .select("*")
            .eq("meeting_id", session_id)
            .limit(1)
            .execute()
        )
        if m.data:
            return m.data[0]
    except Exception as e:
        print(f"[sb] get meeting error: {e}")
    return {}

def _sb_upd_meeting_transcript(session_id: str, new_transcript: str) -> None:
    if not SUPABASE_ENABLED:
        return
    try:
        supabase().table("meetings").update({"transcript": new_transcript}).eq("meeting_id", session_id).execute()
    except Exception as e:
        print(f"[sb] update transcript error: {e}")

def _append_transcript_line(session_id: str, speaker_name: str, text: str) -> None:
    mtg = _sb_get_meeting(session_id)
    prev = mtg.get("transcript") or ""
    line = f"{speaker_name}: {text}".strip()
    new = (prev.rstrip() + "\n" + line) if prev.strip() else line
    _sb_upd_meeting_transcript(session_id, new)

# transcript → pseudo turns (fallback/only path now)
def _parse_transcript_to_turns(transcript_text: str, interviewer_name: str, candidate_name: str) -> List[dict]:
    if not (transcript_text or "").strip():
        return []
    out: List[dict] = []
    lines = [ln.strip() for ln in transcript_text.splitlines() if ln.strip()]
    for ln in lines:
        if ":" in ln:
            name, msg = ln.split(":", 1)
            name = name.strip(); msg = msg.strip()
        else:
            if out:
                out[-1]["text"] += (" " + ln); continue
            name, msg = "Unknown", ln

        role = "unknown"
        if interviewer_name and name.lower() == interviewer_name.lower():
            role = "interviewer"
        elif candidate_name and name.lower() == candidate_name.lower():
            role = "candidate"
        else:
            if name.lower() in ("interviewer", "mic"):
                role = "interviewer"
            elif name.lower() in ("candidate", "tab"):
                role = "candidate"

        out.append({
            "source": "mic" if role == "interviewer" else ("tab" if role == "candidate" else "unknown"),
            "speaker": role,
            "speaker_name": name,
            "text": msg,
            "is_final": True,
        })
    return out

# ─────────────────────────────────────────────────────────────
# Formatting helpers — make outputs point-wise, no asterisks
# ─────────────────────────────────────────────────────────────
_CODE_FENCE = re.compile(r"```.*?```", flags=re.S)

def _strip_code_and_json(s: str) -> str:
    if not s:
        return ""
    s = _CODE_FENCE.sub("", s)
    s = s.replace("`", " ")
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"_+", "", s)
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            s = (obj.get("explanation") or obj.get("expected_answer") or "")
        except Exception:
            s = ""
    return s.strip()

def _to_bullets(s: str, max_items: int = 6, max_len: int = 180) -> str:
    s = _strip_code_and_json(s)
    if not s:
        return ""
    parts = re.split(r"[\r\n]+|(?:^|[\s])[-–•·]\s+", s)
    parts = [p.strip(" -–•·\t\r\n") for p in parts if p and p.strip()]
    if len(parts) <= 1:
        parts = re.split(r"(?<=[.!?])\s+", s)
    clean = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        if len(p) > max_len:
            p = p[:max_len-1].rstrip() + "…"
        clean.append(p)
        if len(clean) >= max_items:
            break
    if not clean:
        return ""
    return "\n".join(f"• {p}" for p in clean)

# ─────────────────────────────────────────────────────────────
# Candidate answer extraction (transcript-only)
# ─────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _is_interviewer(t: dict, interviewer_name: str) -> bool:
    who = (t.get("speaker") or t.get("source") or "").lower()
    name = (t.get("speaker_name") or "").lower()
    if "interviewer" in who or "mic" in who:
        return True
    if interviewer_name and name and name == interviewer_name.lower():
        return True
    return False

def _is_candidate(t: dict, candidate_name: str) -> bool:
    who = (t.get("speaker") or t.get("source") or "").lower()
    name = (t.get("speaker_name") or "").lower()
    if "candidate" in who or "tab" in who:
        return True
    if candidate_name and name and name == candidate_name.lower():
        return True
    return False

def _recent_turns(session_id: str, limit: int = 200) -> List[dict]:
    mtg = _sb_get_meeting(session_id)
    turns = _parse_transcript_to_turns(
        mtg.get("transcript") or "",
        mtg.get("interviewer_name") or "",
        mtg.get("candidate_name") or "",
    )
    if limit and len(turns) > limit:
        return turns[-limit:]
    return turns

def _turns_snippet(turns: List[dict], last_n: int = 12) -> str:
    ctx = turns[-last_n:] if len(turns) > last_n else turns[:]
    lines = []
    for t in ctx:
        who = t.get("speaker_name") or t.get("speaker") or t.get("source", "unknown")
        text = t.get("text", "")
        lines.append(f"{who}: {text}")
    return "\n".join(lines)

def _extract_candidate_answer(question_text: str, turns: List[dict], interviewer_name: str = "", candidate_name: str = "") -> str:
    if not turns:
        return ""
    qn = _norm(question_text)
    best_idx, best_score = None, 0.0
    qtoks = set(qn.split())
    for i, t in enumerate(turns):
        if not _is_interviewer(t, interviewer_name):
            continue
        tn = _norm(t.get("text", ""))
        if not tn:
            continue
        ttoks = set(tn.split())
        if not qtoks or not ttoks:
            continue
        overlap = len(qtoks & ttoks) / max(1, len(qtoks))
        if qn in tn or tn in qn:
            overlap = 1.0
        if overlap > best_score:
            best_score, best_idx = overlap, i

    if best_idx is not None:
        out = []
        for j in range(best_idx + 1, len(turns)):
            t = turns[j]
            if _is_interviewer(t, interviewer_name):
                break
            if _is_candidate(t, candidate_name):
                out.append(t.get("text", ""))
            if len(out) >= 6:
                break
        return " ".join(x.strip() for x in out if x.strip())

    cand = [t.get("text", "") for t in turns[-10:] if _is_candidate(t, candidate_name)]
    return " ".join(x.strip() for x in cand if x.strip())

# ─────────────────────────────────────────────────────────────
# Summary payload coercer — ALWAYS return requested JSON shape
# ─────────────────────────────────────────────────────────────
def _coerce_summary_payload(model_out: Any) -> dict:
    out = {
        "candidate_summary": "",
        "interview_summary": "",
        "strengths": [],
        "risks": [],
        "recommended_next_questions": [],
    }

    if isinstance(model_out, str):
        out["candidate_summary"] = model_out.strip()
        return out

    if not isinstance(model_out, dict):
        return out

    if "raw" in model_out and not any(
        k in model_out for k in ("candidate_summary", "interview_summary", "strengths", "risks", "recommended_next_questions")
    ):
        out["candidate_summary"] = str(model_out.get("raw", "")).strip()
        return out

    out["candidate_summary"] = str(model_out.get("candidate_summary") or model_out.get("raw") or "").strip()
    out["interview_summary"] = str(model_out.get("interview_summary") or "").strip()

    def _as_list(v):
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            parts = [p.strip(" -•\t") for p in v.splitlines() if p.strip()]
            return parts
        return []

    out["strengths"] = _as_list(model_out.get("strengths"))
    out["risks"] = _as_list(model_out.get("risks"))
    out["recommended_next_questions"] = _as_list(model_out.get("recommended_next_questions"))
    return out

# ─────────────────────────────────────────────────────────────
# AI endpoints (kept)
# ─────────────────────────────────────────────────────────────
@app.post("/ai/summary")
async def ai_summary(payload: Dict[str, str] = Body(...)):
    session_id = (payload or {}).get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    meeting = _sb_get_meeting(session_id)
    turns = _recent_turns(session_id, limit=2000)
    resume = meeting.get("resume", "")
    jd     = meeting.get("jd", "")
    lines = [f"{(t.get('speaker_name') or t.get('speaker') or t.get('source'))}: {t.get('text','')}" for t in turns]
    convo = "\n".join(lines)
    prompt = f"""
Return STRICT JSON with:
candidate_summary, interview_summary, strengths (array), risks (array), recommended_next_questions (array).

RESUME:
{resume}

JD:
{jd}

FULL TRANSCRIPT:
{convo}
""".strip()
    data = _ollama_chat_json(prompt)
    coerced = _coerce_summary_payload(data)

    try:
        _sb_upsert_ai_report(session_id, coerced)
    except Exception as e:
        print(f"[ai_summary] store report error: {e}")

    return {"session_id": session_id, **coerced}

@app.post("/ai/expected")
async def ai_expected(payload: Dict[str, str] = Body(...)):
    session_id = (payload or {}).get("session_id")
    question   = (payload or {}).get("question", "").strip()
    if not session_id or not question:
        raise HTTPException(400, "session_id and question required")
    meeting = _sb_get_meeting(session_id)
    resume = meeting.get("resume", "")
    jd     = meeting.get("jd", "")
    turns  = _recent_turns(session_id, 80)
    convo  = _turns_snippet(turns, 12)
    prompt = f"""
Return STRICT JSON with key: expected_answer.

Context:
RESUME:
{resume}

JD:
{jd}

RECENT TRANSCRIPT:
{convo}

QUESTION TO PREPARE EXPECTED ANSWER FOR:
{question}

Write a concise but complete expected answer (bullets or short paragraphs).
""".strip()
    data = _ollama_chat_json(prompt)
    exp_raw = (data.get("expected_answer") or data.get("raw") or "").strip()
    return {
        "session_id": session_id,
        "question": question,
        "expected_answer": _to_bullets(exp_raw, max_items=6),
    }

@app.post("/ai/validate")
async def ai_validate(payload: Dict[str, str] = Body(...)):
    session_id = (payload or {}).get("session_id")
    question   = (payload or {}).get("question", "").strip()
    if not session_id or not question:
        raise HTTPException(400, "session_id and question required")
    meeting = _sb_get_meeting(session_id)
    resume = meeting.get("resume", "")
    jd     = meeting.get("jd", "")
    turns  = _recent_turns(session_id, 300)

    cand_answer = _extract_candidate_answer(
        question,
        turns,
        interviewer_name=(meeting.get("interviewer_name") or ""),
        candidate_name=(meeting.get("candidate_name") or ""),
    )

    prompt = f"""
You must return STRICT JSON with keys:
- verdict (one of: "RIGHT", "ALMOST", "WRONG")
- score (0.0–1.0)
- explanation (1–3 sentences)
- expected_answer
- candidate_answer

Use:
RESUME:
{resume}

JD:
{jd}

QUESTION:
{question}

CANDIDATE_ANSWER:
{cand_answer if cand_answer else "(none)"} 

First, infer the ideal expected_answer from JD/Resume. Then compare candidate_answer to expected_answer and fill verdict/score/explanation.
""".strip()
    data = _ollama_chat_json(prompt)

    verdict = (data.get("verdict") or "").upper().strip()
    if verdict not in {"RIGHT", "ALMOST", "WRONG"}:
        verdict = "WRONG" if not cand_answer else "ALMOST"
    try:
        score = float(data.get("score", 0.0))
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))
    score_pct = int(round(score * 100))

    exp_raw = (data.get("expected_answer") or "").strip()
    explanation_raw = (data.get("explanation") or data.get("raw") or "").strip()

    return {
        "session_id": session_id,
        "question": question,
        "expected_answer": _to_bullets(exp_raw, max_items=6),
        "candidate_answer": cand_answer,
        "verdict": verdict,
        "score": score,
        "score_pct": score_pct,
        "explanation": _to_bullets(explanation_raw, max_items=3, max_len=160),
    }

@app.post("/ai/questions")
async def ai_questions(payload: Dict[str, Any] = Body(...)):
    session_id = (payload or {}).get("session_id")
    count = int((payload or {}).get("count", 5))
    if not session_id:
        raise HTTPException(400, "session_id required")
    meeting = _sb_get_meeting(session_id)
    resume = meeting.get("resume", "")
    jd     = meeting.get("jd", "")
    turns  = _recent_turns(session_id, 60)
    convo  = _turns_snippet(turns, 10)
    prompt = f"""
Return STRICT JSON with key: questions (array of {count} concise follow-up questions).
Use JD and Resume as primary guidance; if transcript context exists, adapt to it; otherwise generate from JD/Resume only.

RESUME:
{resume}

JD:
{jd}

RECENT TRANSCRIPT:
{convo if convo.strip() else "(none)"}
""".strip()
    data = _ollama_chat_json(prompt, temperature=0.6)
    questions = data.get("questions")
    if not isinstance(questions, list):
        raw = (data.get("raw") or "").strip()
        questions = [x.strip("-• ").strip() for x in raw.split("\n") if x.strip()]
    questions = [q for q in questions if q][:count]
    return {"session_id": session_id, "questions": questions}

# ─────────────────────────────────────────────────────────────
# Store AI summary into meetings.ai_report (jsonb column)
# ─────────────────────────────────────────────────────────────
def _sb_upsert_ai_report(session_id: str, report: dict) -> None:
    if not SUPABASE_ENABLED:
        return
    try:
        safe_report = json.loads(json.dumps(report, ensure_ascii=False))
    except Exception as e:
        print(f"[sb] ai_report not JSON-serializable: {e}")
        return
    try:
        supabase().table("meetings").update({"ai_report": safe_report}).eq("meeting_id", session_id).execute()
    except Exception as e:
        print(f"[sb] update meetings.ai_report error: {e}")

# ─────────────────────────────────────────────────────────────
# Uvicorn entrypoint
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Prefer running via CLI: uvicorn server2:app --host 127.0.0.1 --port 8000 --reload
    uvicorn.run("server2:app", host="127.0.0.1", port=8000, reload=True)
