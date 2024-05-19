package com.example.heartprediction_backend.service;

import com.example.heartprediction_backend.dataSource.model.Prediction;
import com.example.heartprediction_backend.dataSource.model.User;
import com.example.heartprediction_backend.dataSource.repository.PredictionRepository;
import com.example.heartprediction_backend.dataSource.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class AddUser implements CommandLineRunner {
    private final PasswordEncoder passwordEncoder;
    private final UserRepository userRepository;
    private final PredictionRepository predictionRepository;

    @Override
    public void run(String... args) throws Exception {
        String email = "ilinca@gmail.com";
        String pass = "bujori";
        String fullName = "marian";

        User user = User.builder()
                .fullName(fullName)
                .email(email)
                .password(passwordEncoder.encode(pass))
                .build();
        userRepository.saveAll(List.of(user));

        Prediction prediction = Prediction.builder()
                .age(40)
                .sex("M")
                .chestPainType("ATA")
                .restingBP(140)
                .cholesterol(289)
                .fastingBS(0)
                .restingECG("Normal")
                .maxHR(172)
                .exerciseAngina("N")
                .oldPeak(0)
                .stSlope("Up")
                .user(user)
                .modelPredictionNormal(0.8906442)
                .modelPredictionDisease(0.1093558)
                .build();

        predictionRepository.save(prediction);
    }
}