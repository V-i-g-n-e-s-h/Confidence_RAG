from client.tk_app import RagConfidenceDocApp

if __name__ == "__main__":
    app = RagConfidenceDocApp(backend_url="http://localhost:8000")
    app.mainloop()