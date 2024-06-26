package br.deeplearning4java.neuralnetwork.database;

import br.deeplearning4java.neuralnetwork.core.activation.Activation;
import br.deeplearning4java.neuralnetwork.core.activation.IActivation;

import javax.persistence.AttributeConverter;
import javax.persistence.Converter;

@Converter
public class ActivationConverter implements AttributeConverter<IActivation, String> {

    @Override
    public String convertToDatabaseColumn(IActivation attribute) {
        if (attribute != null) {
            return attribute.getClass().getSimpleName().toLowerCase();
        }
        return null;
    }

    @Override
    public IActivation convertToEntityAttribute(String dbData) {
        if (dbData != null) {
            return Activation.create(dbData);
        }
        return null;
    }
}