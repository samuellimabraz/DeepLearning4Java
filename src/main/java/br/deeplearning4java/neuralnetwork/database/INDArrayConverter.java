package br.deeplearning4java.neuralnetwork.database;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.persistence.AttributeConverter;
import javax.persistence.Converter;
import javax.persistence.PostLoad;
import javax.persistence.PrePersist;

@Converter
public class INDArrayConverter implements AttributeConverter<INDArray, byte[]> {

    @Override
    @PrePersist
    public byte[] convertToDatabaseColumn(INDArray attribute) {
        if (attribute != null) {
            return attribute.data().asBytes();
        }
        return null;
    }

    @Override
    @PostLoad
    public INDArray convertToEntityAttribute(byte[] dbData) {
        if (dbData != null) {
            return Nd4j.fromByteArray(dbData);
        }
        return null;
    }
}
