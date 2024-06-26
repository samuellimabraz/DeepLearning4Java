package br.deeplearning4java.game.model.database;

import com.sun.jna.platform.win32.OaIdl;

import javax.persistence.EntityManager;
import javax.persistence.EntityTransaction;
import javax.persistence.Persistence;
import java.util.List;

public class PersistenceManager {

    public static void persist(EntityManager entityManager, Object object) {
        EntityTransaction transaction = entityManager.getTransaction();
        try {
            transaction.begin();
            entityManager.persist(object);
            transaction.commit();
            System.out.println("Persisted object: " + object);
        } finally {
            if (transaction.isActive()) {
                transaction.rollback();
            }
        }
    }

    public static void persistAll(EntityManager entityManager, List<Object> objects) {
        EntityTransaction transaction = entityManager.getTransaction();
        try {
            transaction.begin();
            for (Object object : objects) {
                entityManager.persist(object);
                System.out.println("Persisted object: " + object);
            }
            transaction.commit();
        } finally {
            if (transaction.isActive()) {
                transaction.rollback();
            }
        }
    }

    public static EntityManager createEntityManager(String persistenceUnit) {
        return Persistence.createEntityManagerFactory(persistenceUnit).createEntityManager();
    }
}
