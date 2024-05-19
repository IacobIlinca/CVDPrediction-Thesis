package com.example.heartprediction_backend.controller.authentication;


import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class AuthenticationRequestBody {
    @NotBlank(message = "Email shouldn't be blank")
    @Size(min = 1, max = 100)
    private final String userEmail;
    @NotBlank(message = "Password shouldn't be blank")
    @Size(min = 4, max = 40)
    private final String password;
}
