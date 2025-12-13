import queue
import threading
import tkinter as tk
from tkinter import (ttk, messagebox, filedialog,)

import requests


class RagConfidenceDocApp(tk.Tk):
    def __init__(self, backend_url: str = "http://localhost:8000"):
        super().__init__()
        self.title("RAG QA with Confidence and Document Manager")
        self.geometry("1920x1080")
        self.minsize(1000, 700)

        self.backend_url = backend_url
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        self.style.configure("TButton", padding=6)
        self.style.configure("TLabel", padding=3)

        self.worker_q = queue.Queue()

        self._build_ui()
        self._load_docs_async()
        self.after(100, self._poll_worker_queue)

    def _build_ui(self):
        container = ttk.Frame(self, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(container)
        top.pack(fill=tk.X, pady=(0, 8))
        self.status_var = tk.StringVar(value="Connecting to backend…")
        ttk.Label(top, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        q_frame = ttk.Frame(container)
        q_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(q_frame, text="Question", style="Header.TLabel").pack(anchor="w")
        self.q_text = tk.Text(q_frame, height=3, wrap=tk.WORD, font=("Segoe UI", 11))
        self.q_text.pack(fill=tk.X)

        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=tk.X, pady=(8, 8))
        self.ask_btn = ttk.Button(btn_frame, text="Ask", command=self._on_ask)
        self.ask_btn.pack(side=tk.LEFT)
        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self._on_clear)
        self.clear_btn.pack(side=tk.LEFT, padx=(8, 0))

        middle = ttk.Frame(container)
        middle.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(middle)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Answer", style="Header.TLabel").pack(anchor="w")
        self.answer_text = tk.Text(
            left, height=9, wrap=tk.WORD, font=("Segoe UI", 11)
        )
        self.answer_text.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(middle, width=260)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        ttk.Label(
            right, text="Overall confidence", style="Header.TLabel"
        ).pack(anchor="w")
        self.conf_badge = tk.Label(
            right,
            text="—",
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg="#888",
            width=10,
            pady=6,
        )
        self.conf_badge.pack(pady=(4, 10))
        self.conf_label = ttk.Label(
            right, text="", wraplength=240, justify=tk.LEFT
        )
        self.conf_label.pack(anchor="w")

        ttk.Label(
            container,
            text="Retrieved context (per-passage confidence)",
            style="Header.TLabel",
        ).pack(anchor="w", pady=(10, 4))
        cols = ("source", "confidence", "qdrant_score", "preview")
        self.table = ttk.Treeview(
            container, columns=cols, show="headings", height=4
        )
        self.table.heading("source", text="Source")
        self.table.heading("confidence", text="Confidence")
        self.table.heading("qdrant_score", text="Qdrant score")
        self.table.heading("preview", text="Preview")

        self.table.column("source", width=200, anchor=tk.W)
        self.table.column("confidence", width=100, anchor=tk.W)
        self.table.column("qdrant_score", width=100, anchor=tk.W)
        self.table.column("preview", width=500, anchor=tk.W)
        self.table.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="Documents (indexed in Qdrant)",
            style="Header.TLabel",
        ).pack(anchor="w", pady=(10, 4))
        doc_frame = ttk.Frame(container)
        doc_frame.pack(fill=tk.BOTH, expand=True)

        doc_cols = ("doc_id", "name", "path", "chunks")
        self.doc_table = ttk.Treeview(
            doc_frame, columns=doc_cols, show="headings", height=6
        )
        self.doc_table.heading("doc_id", text="Doc ID")
        self.doc_table.heading("name", text="Name")
        self.doc_table.heading("path", text="Path")
        self.doc_table.heading("chunks", text="Chunks")

        self.doc_table.column("doc_id", width=200, anchor=tk.W)
        self.doc_table.column("name", width=200, anchor=tk.W)
        self.doc_table.column("path", width=400, anchor=tk.W)
        self.doc_table.column("chunks", width=80, anchor=tk.E)

        self.doc_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        doc_btns = ttk.Frame(doc_frame)
        doc_btns.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))

        self.upload_btn = ttk.Button(
            doc_btns, text="Upload documents…", command=self._on_upload_docs
        )
        self.upload_btn.pack(fill=tk.X, pady=(0, 6))

        self.view_doc_btn = ttk.Button(
            doc_btns, text="View selected", command=self._on_view_doc
        )
        self.view_doc_btn.pack(fill=tk.X, pady=(0, 6))

        self.delete_doc_btn = ttk.Button(
            doc_btns, text="Delete selected", command=self._on_delete_doc
        )
        self.delete_doc_btn.pack(fill=tk.X, pady=(0, 6))

        self.refresh_docs_btn = ttk.Button(
            doc_btns, text="Refresh list", command=self._on_refresh_docs
        )
        self.refresh_docs_btn.pack(fill=tk.X, pady=(0, 6))

    def _load_docs_async(self):
        def work():
            try:
                resp = requests.get(f"{self.backend_url}/documents", timeout=30)
                resp.raise_for_status()
                docs = resp.json()
                self.worker_q.put(("docs_list_ok", docs))
            except Exception as e:
                self.worker_q.put(("load_err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_ask(self):
        question = self.q_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showerror("Missing question", "Please enter a question.")
            return
        self._set_busy(True, "Retrieving context and generating answer…")

        def work():
            try:
                resp = requests.post(
                    f"{self.backend_url}/qa",
                    json={"question": question},
                    timeout=300,
                )
                resp.raise_for_status()
                result = resp.json()
                self.worker_q.put(("answer_ok", result))
            except Exception as e:
                self.worker_q.put(("answer_err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_clear(self):
        self.q_text.delete("1.0", tk.END)
        self.answer_text.delete("1.0", tk.END)
        self.conf_badge.config(text="—", bg="#888")
        self.conf_label.config(text="")
        for row in self.table.get_children():
            self.table.delete(row)
        self.status_var.set("Cleared.")

    def _on_upload_docs(self):
        paths = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[
                ("Text / Markdown / CSV / Log", "*.txt *.md *.csv *.log"),
                ("PDF", "*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self._set_busy(True, f"Indexing {len(paths)} document(s)…")

        def work():
            try:
                resp = requests.post(
                    f"{self.backend_url}/documents/index",
                    json={"filepaths": list(paths)},
                    timeout=600,
                )
                resp.raise_for_status()
                docs = resp.json()
                self.worker_q.put(("docs_list_ok", docs))
            except Exception as e:
                self.worker_q.put(("docs_err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_refresh_docs(self):
        self._set_busy(True, "Refreshing documents…")

        def work():
            try:
                resp = requests.get(f"{self.backend_url}/documents", timeout=60)
                resp.raise_for_status()
                docs = resp.json()
                self.worker_q.put(("docs_list_ok", docs))
            except Exception as e:
                self.worker_q.put(("docs_err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _get_selected_doc_id(self):
        sel = self.doc_table.selection()
        if not sel:
            return None, None
        item_id = sel[0]
        values = self.doc_table.item(item_id, "values")
        if not values:
            return None, None
        doc_id = values[0]
        name = values[1]
        return doc_id, name

    def _on_view_doc(self):
        doc_id, name = self._get_selected_doc_id()
        if not doc_id:
            messagebox.showinfo(
                "No selection", "Please select a document from the list."
            )
            return
        self._set_busy(True, f"Loading document: {name}…")

        def work():
            try:
                resp = requests.get(
                    f"{self.backend_url}/documents/{doc_id}/text", timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("text", "")
                self.worker_q.put(("doc_view_ok", (name, text)))
            except Exception as e:
                self.worker_q.put(("docs_err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_delete_doc(self):
        doc_id, name = self._get_selected_doc_id()
        if not doc_id:
            messagebox.showinfo(
                "No selection", "Please select a document to delete."
            )
            return
        if not messagebox.askyesno(
            "Confirm delete",
            f"Delete document '{name}' and its chunks from Qdrant?",
        ):
            return

        self._set_busy(True, f"Deleting document: {name}…")

        def work():
            try:
                resp = requests.post(
                    f"{self.backend_url}/documents/delete",
                    json={"doc_id": doc_id},
                    timeout=120,
                )
                resp.raise_for_status()
                docs = resp.json()
                self.worker_q.put(("docs_list_ok", docs))
            except Exception as e:
                self.worker_q.put(("docs_err", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _set_busy(self, busy, msg=None):
        state = "disabled" if busy else "normal"
        for btn in [
            self.ask_btn,
            self.clear_btn,
            self.upload_btn,
            self.view_doc_btn,
            self.delete_doc_btn,
            self.refresh_docs_btn,
        ]:
            btn.config(state=state)
        if msg:
            self.status_var.set(msg)
        self.update_idletasks()

    def _update_conf_badge(self, score):
        pct = score * 100.0
        text = f"{pct:.1f}%"
        if pct < 40:
            color = "#d9534f"
            desc = "Low confidence: answer may be unreliable."
        elif pct < 60:
            color = "#f0ad4e"
            desc = "Medium confidence: treat with some caution."
        else:
            color = "#5cb85c"
            desc = "High confidence: answer is likely reliable."
        self.conf_badge.config(text=text, bg=color)
        self.conf_label.config(text=desc)

    def _populate_sources(self, passages):
        for row in self.table.get_children():
            self.table.delete(row)
        for p in passages:
            src = p.get("source", "unknown")
            conf = float(p.get("confidence", 0.0)) * 100.0
            qdr = float(p.get("qdrant_score", 0.0))
            text = (p.get("text", "") or "").replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "..."
            self.table.insert(
                "",
                tk.END,
                values=(src, f"{conf:.1f}%", f"{qdr:.4f}", text),
            )

    def _populate_docs(self, docs):
        for row in self.doc_table.get_children():
            self.doc_table.delete(row)
        for d in docs:
            self.doc_table.insert(
                "",
                tk.END,
                values=(
                    d.get("doc_id", ""),
                    d.get("name", ""),
                    d.get("path", ""),
                    d.get("chunks", 0),
                ),
            )
        self.status_var.set(f"{len(docs)} document(s) indexed.")

    def _poll_worker_queue(self):
        try:
            while True:
                tag, payload = self.worker_q.get_nowait()
                if tag == "load_err":
                    self.status_var.set("Error connecting to backend.")
                    messagebox.showerror("Initialization error", payload)
                elif tag == "answer_ok":
                    self._set_busy(False, "Done.")
                    answer = payload.get("answer", "")
                    overall = float(payload.get("overall_confidence", 0.0))
                    passages = payload.get("passages", [])
                    self.answer_text.delete("1.0", tk.END)
                    self.answer_text.insert(tk.END, answer)
                    self._update_conf_badge(overall)
                    self._populate_sources(passages)
                elif tag == "answer_err":
                    self._set_busy(False, "Error.")
                    messagebox.showerror("Answer error", payload)
                elif tag == "docs_list_ok":
                    self._set_busy(False, "Done.")
                    self._populate_docs(payload)
                elif tag == "docs_err":
                    self._set_busy(False, "Error.")
                    messagebox.showerror("Document error", payload)
                elif tag == "doc_view_ok":
                    self._set_busy(False, "Done.")
                    name, text = payload
                    self._show_document_view(name, text)
        except queue.Empty:
            pass
        self.after(100, self._poll_worker_queue)

    def _show_document_view(self, name, text):
        win = tk.Toplevel(self)
        win.title(f"View document: {name}")
        win.geometry("800x600")
        txt = tk.Text(win, wrap=tk.WORD, font=("Segoe UI", 11))
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, text)
        txt.config(state="disabled")
