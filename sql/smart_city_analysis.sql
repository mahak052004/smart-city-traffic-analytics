CREATE SCHEMA IF NOT EXISTS smart_city;


CREATE TABLE IF NOT EXISTS smart_city.sensor_data (

    sensor_id VARCHAR(50),

    location VARCHAR(100),

    date_time TIMESTAMP,

    vehicle_count INTEGER,

    average_speed NUMERIC(10,2),

    congestion_level VARCHAR(30)

);


CREATE TABLE IF NOT EXISTS smart_city.accident_data (

    accident_id VARCHAR(50),

    date_time TIMESTAMP,

    location VARCHAR(100),

    weather_condition VARCHAR(50),

    road_condition VARCHAR(50),

    vehicle_type VARCHAR(50),

    accident_severity VARCHAR(30),

    number_of_vehicles INTEGER,

    casualties INTEGER,

    traffic_density VARCHAR(30)

);



SELECT

    location,

    COUNT(*) AS observations,

    ROUND(
        AVG(vehicle_count),
        2
    ) AS avg_vehicle_count,

    ROUND(
        AVG(average_speed),
        2
    ) AS avg_speed,

    ROUND(

        100.0 *

        SUM(
            CASE
                WHEN congestion_level = 'High'
                THEN 1
                ELSE 0
            END
        )
        /
        COUNT(*),

        2

    ) AS high_congestion_rate

FROM smart_city.sensor_data

GROUP BY location

ORDER BY avg_vehicle_count DESC;



SELECT

    EXTRACT(
        HOUR FROM date_time
    ) AS hour,

    ROUND(
        AVG(vehicle_count),
        2
    ) AS avg_vehicle_count,

    ROUND(
        AVG(average_speed),
        2
    ) AS avg_speed

FROM smart_city.sensor_data

GROUP BY

    EXTRACT(
        HOUR FROM date_time
    )

ORDER BY hour;




SELECT

    location,

    COUNT(*) AS accident_count,

    SUM(casualties) AS total_casualties,

    ROUND(
        AVG(casualties),
        2
    ) AS avg_casualties

FROM smart_city.accident_data

GROUP BY location

ORDER BY accident_count DESC;



SELECT

    accident_severity,

    COUNT(*) AS accident_count,

    ROUND(

        100.0 *
        COUNT(*)
        /
        SUM(COUNT(*))
        OVER(),

        2

    ) AS percentage

FROM smart_city.accident_data

GROUP BY accident_severity

ORDER BY accident_count DESC;


SELECT

    weather_condition,

    COUNT(*) AS accident_count,

    SUM(casualties) AS total_casualties

FROM smart_city.accident_data

GROUP BY weather_condition

ORDER BY accident_count DESC;




SELECT

    road_condition,

    COUNT(*) AS accident_count,

    SUM(casualties) AS casualties

FROM smart_city.accident_data

GROUP BY road_condition

ORDER BY accident_count DESC;




SELECT

    vehicle_type,

    COUNT(*) AS accident_count,

    SUM(casualties) AS casualties

FROM smart_city.accident_data

GROUP BY vehicle_type

ORDER BY accident_count DESC;


SELECT

    traffic_density,

    COUNT(*) AS accident_count,

    SUM(casualties) AS casualties

FROM smart_city.accident_data

GROUP BY traffic_density

ORDER BY accident_count DESC;