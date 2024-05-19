package com.example.heartprediction_backend.dataSource.model;

import jakarta.persistence.*;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@Entity
@Table(name = "Predictions")
@SuperBuilder
@RequiredArgsConstructor
public class Prediction {
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private int predictionId;
    @ManyToOne(fetch = FetchType.LAZY)
    private User user;

    @Basic(optional = false)
    private int age;

    @Basic(optional = false)
    private String sex;

    @Basic(optional = false)
    private String chestPainType;

    @Basic(optional = false)
    private int restingBP;

    @Basic(optional = false)
    private int cholesterol;

    @Basic(optional = false)
    private int fastingBS;

    @Basic(optional = false)
    private String  restingECG;

    @Basic(optional = false)
    private int maxHR;

    @Basic(optional = false)
    private String exerciseAngina;

    @Basic(optional = false)
    private float oldPeak;

    @Basic(optional = false)
    private String stSlope;

    @Basic(optional = false)
    private Double modelPredictionNormal;

    @Basic(optional = false)
    private Double modelPredictionDisease;

}
