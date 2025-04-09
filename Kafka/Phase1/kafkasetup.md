**Log of Kafka Installation - That worked!**

This was the most time intensive step (took about 2 days! just to get Kafka started). The steps below is what I now follow to get Kafka running.
Prior to this, I fixed file path in my .sh file to make it bash compatible.

1. Clear all logs (from tmp folder outside the main folder, and from the logs inside the kafka folder) 
2. In a bash window: cd D:/kafka_2.13-4.0.0
3. Create new meta properties file: bin/kafka-storage.sh format -t random -c config/kraft/server.properties --standalone
4. bin/kafka-server-start.sh config/server.properties
5. Create kafka topic if not created

NEW BASH WINDOW
1. cd D:/Grace/CMU/Courses/Spring2025/OpAI/Assignment/1/
2. python producer.py

NEW BASH WINDOW
1. cd D:/Grace/CMU/Courses/Spring2025/OpAI/Assignment/1/
2. python consumer.py


------------------------------------------------------------------------------------------------------------------------------------------------------

**Log of Kafka Installation/Setup Issues**

**1. Initial Setup & Configuration**
**Query**: Steps to start Kafka server on Windows (KRaft mode) via PowerShell.   
**Solution**:  
- Install Java and Kafka.  
- Configure `server.properties` for KRaft mode.  
- Format storage with a UUID.  
- Start Kafka and test with topic creation/producer/consumer.  


**2. Server Properties Validation**
**Query**: "Is this KRaft server properties okay?"  
**Shared Configuration**:  
- `process.roles=broker,controller`  
- `controller.quorum.voters=1@localhost:9093`  
- Missing `listener.security.protocol.map`.  
**Solution**:  
- Revised `server.properties` to include:  
  ```properties
  listener.security.protocol.map=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
  ```
- Validated paths (`log.dirs=C:/kafka/tmp/kafka-logs`).  


**3. Storage Formatting Error**
**User Query**: Error during storage formatting:  
```powershell
.\bin\windows\kafka-storage.bat format -t uuidgen -c .\config\server.properties
ERROR: No configuration found for '266474c2' at 'null' in 'null'
```
**Root Cause**:  
- Incorrect use of `uuidgen` (UUID not generated).  
- Possible misconfigured `server.properties` or invalid paths.  
**Solution**:  
- Generate UUID via PowerShell: `[guid]::NewGuid().ToString()`.  
- Ensure `server.properties` includes required KRaft settings.  
- Verify `log.dirs` path exists.  


**4. Persistent Storage Error with Valid UUID**
**User Query**: Same error after using a valid UUID.  
**Error**:  
```powershell
.\kafka-storage.bat format -t 88dc8b2a-3b61-475b-9b42-a46f5a71a24a -c .\config\server.properties
ERROR: No configuration found for '266474c2' at 'null' in 'null'
```
**Root Cause**:  
- Misconfigured `server.properties` (e.g., missing `controller.quorum.voters`).  
- Invalid file encoding (e.g., BOM in `server.properties`).  
**Solution**:  
- Provided a complete `server.properties` template for KRaft.  
- Ensured `log.dirs=C:/kafka/tmp/kafka-logs` was valid.  
- Advised creating the log directory: `mkdir C:\kafka\tmp\kafka-logs`.  


**5. Version Check Error**
**Query**: Error when running `kafka-topics.bat --version`:  
```powershell
ERROR: No configuration found for '266474c2' at 'null' in 'null'
4.0.0
```
**Root Cause**:  
- Kafka 4.0 requires KRaft mode but couldn’t find valid cluster metadata.  
- Storage not formatted or misconfigured `server.properties`.  
**Solution**:  
- Reiterated the need to format storage with a UUID.  
- Validated Java 17+ installation.  
- Provided debug steps (check logs, file permissions, reinstall Kafka).  


**6. Final Debugging Steps**
**Query**: All steps followed but error persists.  
**Solution**:  
- **Reinstall Kafka**: Start fresh to eliminate corrupted files.  
- **File Encoding**: Ensure `server.properties` is saved in **UTF-8 without BOM**.  
- **Log Inspection**: Check `C:\kafka\tmp\kafka-logs` for detailed errors.  
- **Permissions**: Grant full write access to `C:\kafka` and subdirectories.  


_Reference_

_Error summary log created using [Perplexity: pplx.ai/share](https://www.perplexity.ai/search/give-me-steps-to-start-my-kafk-lEo3enZZSout2R45kLsL2A)_
