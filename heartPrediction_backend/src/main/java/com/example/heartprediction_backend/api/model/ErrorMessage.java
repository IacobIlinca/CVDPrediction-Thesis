package com.example.heartprediction_backend.api.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@AllArgsConstructor
@RequiredArgsConstructor
public class ErrorMessage {
    private String message;
    private int statusCode;

}