package com.example.heartprediction_backend.api.model;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class PredictionBody {
    @Positive(message = "Age shouldn't be negative")
    private int age;
    @NotBlank(message = "Sex shouldn't be blank")
    private String sex;
    @NotBlank(message = "ChestPainType shouldn't be blank")
    private String chestPainType;
    private int restingBP;
    private int cholesterol;
    private int fastingBS;
    @NotBlank(message = "RestingECG shouldn't be blank")
    private String  restingECG;
    private int maxHR;
    private String exerciseAngina;
    private float oldPeak;
    @NotBlank(message = "StSlope shouldn't be blank")
    private String stSlope;
}