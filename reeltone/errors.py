class BenchError(Exception):
    def __init__(self, message, status=400, code="invalid_request"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


class Absent(BenchError):
    def __init__(self, message):
        super().__init__(message, 404, "not_found")


class Conflict(BenchError):
    def __init__(self, message):
        super().__init__(message, 409, "conflict")


class Limit(BenchError):
    def __init__(self, message):
        super().__init__(message, 413, "limit")
