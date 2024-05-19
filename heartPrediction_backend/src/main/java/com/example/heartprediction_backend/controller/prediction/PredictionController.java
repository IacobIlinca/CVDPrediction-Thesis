package com.example.heartprediction_backend.controller.prediction;

import com.example.heartprediction_backend.api.model.PredictionBody;
import com.example.heartprediction_backend.dataSource.model.Prediction;
import com.example.heartprediction_backend.dataSource.model.PredictionDto;
import com.example.heartprediction_backend.dataSource.model.User;
import com.example.heartprediction_backend.service.PredictionService;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/prediction")
@RequiredArgsConstructor
@Validated
public class PredictionController {
    private final PredictionService predictionService;

    @GetMapping
    @SecurityRequirement(name = "bearerAuthentication")
    public ResponseEntity<List<PredictionDto>> getAllPredictions(
            @AuthenticationPrincipal User user) {
        return ResponseEntity.ok(predictionService.findAllByUser(user));
    }

    @SecurityRequirement(name = "bearerAuthentication")
    @PostMapping
    public ResponseEntity<PredictionDto> predictHeartDisease(
            @AuthenticationPrincipal User principal,
            @Valid @RequestBody PredictionBody predictionBody) {
        var requestDto = predictionService
                .predictHeartDisease(predictionBody, principal);
        return ResponseEntity.ok(requestDto);
    }
}
