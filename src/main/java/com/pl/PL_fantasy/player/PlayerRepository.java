package com.pl.PL_fantasy.player;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface PlayerRepository extends JpaRepository<Player, String> {
    void deleteByWebName(String playerName);

    Optional<Player> findByWebName(String web_name);
}
