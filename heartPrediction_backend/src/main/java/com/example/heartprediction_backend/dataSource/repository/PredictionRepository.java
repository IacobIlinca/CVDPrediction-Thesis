package com.example.heartprediction_backend.dataSource.repository;

import com.example.heartprediction_backend.dataSource.model.Prediction;
import com.example.heartprediction_backend.dataSource.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Integer> {
    List<Prediction> findAllByUser(User user);
}
