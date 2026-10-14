from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        elif self.path == '/version':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"version": "1.0.0"}')
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Path not found")


def run(server_class=HTTPServer, handler_class=SimpleHandler):
    server_address = ('127.0.0.1', 8000) # Bind strictly to localhost and port 8000
    httpd = server_class(server_address, handler_class)
    print("Server running on http://127.0.0.1:8000...")
    httpd.serve_forever() # Keep it running

if __name__ == '__main__':
    run()

