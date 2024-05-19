package com.example.heartprediction_backend.dataSource.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class PredictionDto {
    private int predictionId;
    private int age;
    private String sex;
    private String chestPainType;
    private int restingBP;
    private int cholesterol;
    private int fastingBS;
    private String  restingECG;
    private int maxHR;
    private String exerciseAngina;
    private float oldPeak;
    private String stSlope;
    private Double modelPredictionNormal;
    private Double modelPredictionDisease;

}
