package com.example.heartprediction_backend.exception;

public class InvalidTokenType extends RuntimeException{
    public InvalidTokenType() {
    }

    public InvalidTokenType(String message) {
        super(message);
    }
}

