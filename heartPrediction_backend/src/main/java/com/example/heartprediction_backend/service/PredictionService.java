package com.example.heartprediction_backend.service;

import com.example.heartprediction_backend.api.model.AIModelResponse;
import com.example.heartprediction_backend.api.model.PredictionBody;
import com.example.heartprediction_backend.dataSource.model.Prediction;
import com.example.heartprediction_backend.dataSource.model.PredictionDto;
import com.example.heartprediction_backend.dataSource.model.User;
import com.example.heartprediction_backend.dataSource.repository.PredictionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class PredictionService {
    private final PredictionRepository predictionRepository;

    private final RestTemplate restTemplate;

    public List<PredictionDto> findAllByUser(User user) {
        return predictionRepository.findAllByUser(user).stream().map(this::toPredictionDto).toList();
    }

    public PredictionDto predictHeartDisease(PredictionBody predictionBody, User principal) {

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<PredictionBody> request = new HttpEntity<>(predictionBody, headers);

        String url = "http://localhost:5000/predict";  // URL of the Python Flask service
        ResponseEntity<AIModelResponse> response = restTemplate.postForEntity(url, request, AIModelResponse.class);

        AIModelResponse responseAi = response.getBody();
        Prediction predictionReal = Prediction.builder()
                .user(principal)
                .age(predictionBody.getAge())
                .sex(predictionBody.getSex())
                .chestPainType(predictionBody.getChestPainType())
                .restingBP(predictionBody.getRestingBP())
                .cholesterol(predictionBody.getCholesterol())
                .fastingBS(predictionBody.getFastingBS())
                .restingECG(predictionBody.getRestingECG())
                .maxHR(predictionBody.getMaxHR())
                .exerciseAngina(predictionBody.getExerciseAngina())
                .oldPeak(predictionBody.getOldPeak())
                .stSlope(predictionBody.getStSlope())
                .modelPredictionNormal(responseAi.getPrediction().get(0))
                .modelPredictionDisease(responseAi.getPrediction().get(1))
                .build();

        predictionRepository.save(predictionReal);
        return toPredictionDto(predictionReal);
    }

    private PredictionDto toPredictionDto(Prediction prediction) {
        return PredictionDto.builder()
                .predictionId(prediction.getPredictionId())
                .age(prediction.getAge())
                .sex(prediction.getSex())
                .chestPainType(prediction.getChestPainType())
                .restingBP(prediction.getRestingBP())
                .cholesterol(prediction.getCholesterol())
                .fastingBS(prediction.getFastingBS())
                .restingECG(prediction.getRestingECG())
                .maxHR(prediction.getMaxHR())
                .exerciseAngina(prediction.getExerciseAngina())
                .oldPeak(prediction.getOldPeak())
                .stSlope(prediction.getStSlope())
                .modelPredictionNormal(prediction.getModelPredictionNormal())
                .modelPredictionDisease(prediction.getModelPredictionDisease())
                .build();
    }
}
