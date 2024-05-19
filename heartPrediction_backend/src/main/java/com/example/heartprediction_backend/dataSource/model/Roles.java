package com.example.heartprediction_backend.dataSource.model;

import java.text.MessageFormat;

public enum Roles {
    USER;

    public static String getRoleName(Roles roles) {
        return MessageFormat.format("ROLE_{0}", roles.name());
    }
}
