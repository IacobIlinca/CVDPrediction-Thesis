package com.example.heartprediction_backend.controller.authentication;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class AuthenticationResponseBody {
    private String authenticationToken;
    private String refreshToken;
    private int userId;

    public AuthenticationResponseBody() {
    }
}
