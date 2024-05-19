package com.example.heartprediction_backend.controller.authentication;

import com.example.heartprediction_backend.exception.InvalidTokenType;
import com.example.heartprediction_backend.service.AuthenticationService;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class AuthenticationController {
    private final AuthenticationService authenticationService;

    @PostMapping("/sign-in")
    public ResponseEntity<AuthenticationResponseBody> signInUser(
            @Valid @RequestBody AuthenticationRequestBody authenticationRequestBody) {
        AuthenticationResponseBody body = authenticationService
                .authenticateUser(authenticationRequestBody);
        return ResponseEntity.ok(body);
    }

    @SecurityRequirement(name = "bearerAuthentication")
    @PostMapping("/refreshToken")
    public ResponseEntity<AuthenticationResponseBody> refreshToken(
            @AuthenticationPrincipal UserDetails principal,
            @Valid @RequestBody RefreshTokenRequestBody refreshTokenRequestBody) {
        AuthenticationResponseBody body = authenticationService
                .refreshToken(principal, refreshTokenRequestBody);
        return ResponseEntity.ok(body);
    }

    @ExceptionHandler(BadCredentialsException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String, String> authenticationExceptionHandler(BadCredentialsException badCredentialsException) {
        return Map.of("error", badCredentialsException.getMessage());
    }

    @ExceptionHandler(InvalidTokenType.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String, String> invalidTokenTypeExceptionHandler(InvalidTokenType invalidTokenType) {
        return Map.of("error", invalidTokenType.getMessage());
    }
}
