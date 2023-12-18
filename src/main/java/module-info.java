module br.edu.unifei.ecot12.deeplearning4java {
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

    exports br.edu.unifei.ecot12.deeplearning4java.game.controller;
    opens br.edu.unifei.ecot12.deeplearning4java.game.controller to javafx.fxml;
    exports br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.core.models;
    exports br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.examples.mnist;
    exports br.edu.unifei.ecot12.deeplearning4java.neuralnetwork.examples.qdraw;
}