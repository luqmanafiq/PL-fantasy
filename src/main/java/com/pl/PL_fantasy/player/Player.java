package com.pl.PL_fantasy.player;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name="fantasy")
public class Player {
    @Id
    @Column(name = "web_name", unique = true)
    private String webName;
    private String team;
    private String position;
    private Double cost;
    private int total_points;
    private int goals;
    private int assists;
    private Double minutes;
    private Double ict_index;

    public Player() {
    }

    public Player(String web_name, String team, String position, Double cost, int total_points, int goals, int assists, Double minutes, Double ict_index) {
        this.webName = web_name;
        this.team = team;
        this.position = position;
        this.cost = cost;
        this.total_points = total_points;
        this.goals = goals;
        this.assists = assists;
        this.minutes = minutes;
        this.ict_index = ict_index;
    }

    public String getWebName() {
        return webName;
    }

    public void setWebName(String webName) {
        this.webName = webName;
    }

    public Double getIct_index() {
        return ict_index;
    }

    public void setIct_index(Double ict_index) {
        this.ict_index = ict_index;
    }

    public Double getMinutes() {
        return minutes;
    }

    public void setMinutes(Double minutes) {
        this.minutes = minutes;
    }

    public int getAssists() {
        return assists;
    }

    public void setAssists(int assists) {
        this.assists = assists;
    }

    public int getGoals() {
        return goals;
    }

    public void setGoals(int goals) {
        this.goals = goals;
    }

    public int getTotal_points() {
        return total_points;
    }

    public void setTotal_points(int total_points) {
        this.total_points = total_points;
    }

    public Double getCost() {
        return cost;
    }

    public void setCost(Double cost) {
        this.cost = cost;
    }

    public String getPosition() {
        return position;
    }

    public void setPosition(String position) {
        this.position = position;
    }

    public String getTeam() {
        return team;
    }

    public void setTeam(String team) {
        this.team = team;
    }
}
