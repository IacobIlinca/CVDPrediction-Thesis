package com.example.heartprediction_backend.api.model;

import lombok.Data;

import java.util.List;

@Data
public class AIModelResponse {

    private List<Double> prediction;
}
