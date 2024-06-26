module br.deeplearning4java {
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.web;
    requires javafx.swing;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.ikonli.javafx;
    requires org.kordamp.bootstrapfx.core;

    requires nd4j.api;
    requires opencv;
    requires java.persistence;
    requires java.sql;
    requires org.hibernate.orm.core;
    requires org.mariadb.jdbc;
    requires guava;
    requires com.sun.jna.platform;

    opens br.deeplearning4java.game.model to org.hibernate.orm.core;

    exports br.deeplearning4java.game.controller;
    opens br.deeplearning4java.game.controller to javafx.fxml;
    exports br.deeplearning4java.neuralnetwork.core.models;
    exports br.deeplearning4java.neuralnetwork.examples.imageclassification;
    exports br.deeplearning4java.neuralnetwork.examples.imageclassification.mnist;
    exports br.deeplearning4java.neuralnetwork.examples.imageclassification.qdraw;
    opens br.deeplearning4java.game.model.database to org.hibernate.orm.core;
    opens br.deeplearning4java.neuralnetwork.core.activation to org.hibernate.orm.core;
    opens br.deeplearning4java.neuralnetwork.core.layers to org.hibernate.orm.core;
    opens br.deeplearning4java.neuralnetwork.core.models to org.hibernate.orm.core;
    opens br.deeplearning4java.neuralnetwork.database to org.hibernate.orm.core;
}