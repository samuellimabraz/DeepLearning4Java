module br.deeplearning4java {
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.web;
    requires javafx.swing;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.ikonli.javafx;
    requires org.kordamp.bootstrapfx.core;
    requires guava;
    requires com.sun.jna.platform;

    requires nd4j.api;
    requires opencv;
    requires java.persistence;
    requires java.sql;
    requires org.hibernate.orm.core;
    requires org.mariadb.jdbc;
    requires morphia.core;
    requires org.mongodb.bson;
    requires org.mongodb.driver.core;
    requires org.mongodb.driver.sync.client;
    requires JMathPlot;

    exports br.deeplearning4java.game.controller;
    opens br.deeplearning4java.game.controller to javafx.fxml;
    exports br.deeplearning4java.neuralnetwork.core.models;
    exports br.deeplearning4java.neuralnetwork.core.layers;
    exports br.deeplearning4java.neuralnetwork.examples.classification.image;
    exports br.deeplearning4java.neuralnetwork.examples.classification.image.mnist;
    exports br.deeplearning4java.neuralnetwork.examples.classification.image.qdraw;

    opens br.deeplearning4java.game.model to org.hibernate.orm.core;
    opens br.deeplearning4java.game.model.database to org.hibernate.orm.core;
    opens br.deeplearning4java.neuralnetwork.core.activation to morphia.core;
    opens br.deeplearning4java.neuralnetwork.core.layers to morphia.core;
    opens br.deeplearning4java.neuralnetwork.core.models to morphia.core;
//    opens br.deeplearning4java.neuralnetwork.core.optimizers to morphia.core;
//    opens br.deeplearning4java.neuralnetwork.core.train to morphia.core;
//    opens br.deeplearning4java.neuralnetwork.core.losses to morphia.core;
    opens br.deeplearning4java.neuralnetwork.database to morphia.core;
}