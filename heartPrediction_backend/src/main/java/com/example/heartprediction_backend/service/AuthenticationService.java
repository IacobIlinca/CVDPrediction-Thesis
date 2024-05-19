package com.example.heartprediction_backend.service;

import com.example.heartprediction_backend.controller.authentication.AuthenticationRequestBody;
import com.example.heartprediction_backend.controller.authentication.AuthenticationResponseBody;
import com.example.heartprediction_backend.controller.authentication.RefreshTokenRequestBody;
import com.example.heartprediction_backend.dataSource.model.Roles;
import com.example.heartprediction_backend.dataSource.model.User;
import com.example.heartprediction_backend.dataSource.repository.UserRepository;
import com.example.heartprediction_backend.exception.InvalidTokenType;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;


@Service
@RequiredArgsConstructor
public class AuthenticationService {
    private final UserDetailsService userDetailsService;
    private final UserRepository userRepository;
    private final JwtService jwtService;
    private final AuthenticationManager authenticationManager;
    private static final String ROLE_CLAIM = "role";
    public static final String TOKEN_TYPE_USER = "user";
    private static final String TOKEN_TYPE_REFRESH = "refresh";
    static final String TOKEN_TYPE_NAME = "type";

    public AuthenticationResponseBody authenticateUser(AuthenticationRequestBody authenticationRequestBody) {
        final String userEmail = authenticationRequestBody.getUserEmail();
        final String userPassword = authenticationRequestBody.getPassword();

        authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(userEmail, userPassword)
        );
        UserDetails user = userDetailsService.loadUserByUsername(userEmail);
        return getAuthenticationResponse(user);
    }

    public AuthenticationResponseBody refreshToken(UserDetails principal, RefreshTokenRequestBody refreshTokenRequestBody) {
        String refreshToken = refreshTokenRequestBody.getRefreshToken();
        String tokenType = jwtService.getTokenType(refreshToken);
        if (!TOKEN_TYPE_REFRESH.equals(tokenType) || !jwtService.isTokenValid(refreshToken, principal)) {
            throw new InvalidTokenType("Not a refresh token");
        }

        return getAuthenticationResponse(principal);
    }

    private AuthenticationResponseBody getAuthenticationResponse(UserDetails principal) {
        var extraClaims = new HashMap<String, Object>();
        extraClaims.put(ROLE_CLAIM, getRoleName(principal));
        final String jwt = generateJwtUserToken(extraClaims, principal);
        final String refreshToken = generateJwtRefreshToken(extraClaims, principal);
        User user = (User) principal;

        return AuthenticationResponseBody.builder()
                .authenticationToken(jwt)
                .refreshToken(refreshToken)
                .userId(user.getId())
                .build();
    }

    private Object getRoleName(UserDetails user) {
        SimpleGrantedAuthority userAuthority = (SimpleGrantedAuthority) (user.getAuthorities().toArray()[0]);
        return userAuthority.getAuthority();
    }

    private String generateJwtUserToken(Map<String, Object> extraClaims, UserDetails user) {
        Map<String, Object> extraClaimsCopy = new HashMap<>(extraClaims);
        extraClaimsCopy.put(TOKEN_TYPE_NAME, TOKEN_TYPE_USER);
        return jwtService.generateJwtToken(extraClaimsCopy, user);
    }

    private String generateJwtRefreshToken(Map<String, Object> extraClaims, UserDetails user) {
        Map<String, Object> extraClaimsCopy = new HashMap<>(extraClaims);
        extraClaimsCopy.put(TOKEN_TYPE_NAME, TOKEN_TYPE_REFRESH);
        return jwtService.generateJwtToken(extraClaimsCopy, user);
    }
}
