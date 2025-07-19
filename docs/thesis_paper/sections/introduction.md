# 1. Introduction

## 1.1 Background
Chatbots are increasingly being adopted across various industries to enhance customer service and operational efficiency, with the restaurant sector providing a clear example. For instance, (Gupta, Dheekonda, & Masum, 2024) introduced a chatbot named Genie, which efficiently handles customer inquiries about reservations, menus, and frequently asked questions using predefined responses. While this rule-based approach has proven effective for routine queries, it lacks the flexibility and depth of understanding that large language models (LLMs) offer. However, LLM-based chatbots face a significant limitation: once trained, they cannot automatically incorporate new information and must be retrained with updated datasets to remain accurate and relevant—a process that is both time-consuming and computationally expensive. This challenge can be addressed through Retrieval-Augmented Generation (RAG), an approach that allows LLMs to dynamically retrieve and integrate the most current information from external sources, enabling more adaptive, informative, and contextually relevant responses without the need for constant retraining.

This capability is especially beneficial in industries where information frequently changes and must be tailored to individual cases, such as travel agencies in the Philippines that assist clients with obtaining Japan visas. These agencies manage a high volume of inquiries related to the required documents for different visa types, including tourist, business, and family-related visits. While (Embassy of Japan in the Philippines, 2025) provides general visa requirements on its website, these guidelines are often not exhaustive. Many supporting documents—especially those that serve as proof of the applicant’s specific purpose for visiting—are not publicly listed but are essential for successful applications. For example, if the applicant is visiting Japan to care for a child, they must submit a baby book; if the visit is for medical reasons, a medical certificate is typically required. These specialized documents vary depending on the applicant’s situation and are typically known only to travel agencies, who stay informed through frequent updates and experience.

As a result, clients often seek clarification before visiting the agency to avoid delays or multiple trips due to incomplete submissions. A large number of these inquiries are made via messaging platforms, particularly Facebook Messenger, which is commonly used in the Philippines for casual and business communication. While convenient, manually responding to each message consumes time and resources, especially when questions are repetitive or case-specific. Implementing a RAG-based chatbot system within these messaging platforms would allow travel agencies to automatically provide real-time, accurate, and context-aware answers to client inquiries. This solution would not only reduce the workload of staff but also improve client preparedness, enhance satisfaction, and streamline the overall Japan visa application process.

## 1.2 Problem Statement
As the demand for Japan visas continues to grow in the Philippines, travel agencies have become essential intermediaries in helping applicants navigate the often complex and evolving requirements set by the Japan Embassy. However, despite the availability of general information on the Embassy’s official website, many applicants still struggle to understand what specific documents are needed based on their unique travel purposes. This confusion results in frequent inquiries—particularly through platforms like Facebook Messenger—and inefficiencies in the application process. These challenges highlight several key problems faced by both clients and agencies:

- Incomplete and non-public visa information:
The Japan Embassy in the Philippines provides only general documentation requirements online, while many situational documents—such as baby books for childcare-related visits or medical certificates for health-related travel—are not publicly listed. Travel agencies possess this knowledge through experience and updates, but applicants often lack access to it, leading to confusion and unprepared visits.

- High volume of repetitive, case-specific inquiries via Messenger:
Clients frequently reach out to travel agencies through Facebook Messenger to ask about Japan visa requirements, particularly for unique or changing cases. Responding manually to these repetitive inquiries consumes significant staff time and limits the agency’s ability to focus on more complex tasks.

- Inefficiencies in the visa application process:
Due to a lack of clear, accessible, and real-time information, many applicants arrive at travel agencies with incomplete documents, resulting in delays, repeat visits, and frustration. This creates inefficiencies in both client service and internal agency operations, signaling the need for a more adaptive and automated communication solution.

## 1.3 Objective
This study aims to develop a practical solution that addresses the challenges faced by travel agencies in responding to Japan visa-related inquiries, particularly through messaging platforms like Facebook Messenger. The solution focuses on enhancing service efficiency, improving client satisfaction, and streamlining the visa application process using a chatbot integrated with Retrieval-Augmented Generation (RAG) capabilities.

### 1.3.1 General Objective
To build a prototype chatbot integrated into Facebook Messenger that can answer client inquiries regarding Japan visa requirements, serving as a foundational tool for future development in the travel agency sector aimed at enhancing service efficiency, improving client satisfaction, and streamlining the Japan visa application process.

### 1.3.2 Specific Objectives

1. To collect and create a relevant dataset from one of the travel agencies based on updated Japan visa requirements and real-world inquiries in order to build and evaluate the chatbot’s performance.

2. To design and develop a model that simplifies user queries before retrieving the relevant information.

3. To fine-tune the generator model in the Retrieval-Augmented Generation (RAG) architecture using the collected dataset to improve the quality and relevance of the chatbot's responses.

4. To build a working prototype of the chatbot that effectively answers Japan visa-related inquiries through Messenger, offering real-time, context-aware, and updated responses.

## 1.4 Scope and Limitations

### 1.4.1 Scope of the Study
This study focuses on developing a chatbot prototype that answers inquiries specifically about Japan visa requirements in English, using data gathered from a travel agency branch in Quezon City that actively handles such inquiries, primarily through Facebook Messenger. The chatbot will be integrated exclusively within Messenger, where most client interactions occur. The data sources for training the chatbot include:

- Frequently asked client inquiries provided by the travel agency, based on real-world questions they commonly receive about Japan visa requirements. This includes non-public knowledge the agency has acquired over time from handling various visa applications.

- Official documentation from the Japan Embassy in the Philippines website, particularly from the Visa/Consular Services section, where downloadable PDFs outline the general visa requirements.

The travel agency has granted permission to access past conversation history for research purposes, and all personally identifiable information (PII) will be removed to comply with the Data Privacy Act of 2012 (RA 10173). Although conversations may occur in English, Tagalog, or Cebuano, the study will focus exclusively on English-language interactions. The chatbot will be limited to addressing document-related inquiries concerning Japan visa applications and will serve as a foundational prototype for enhancing customer service in the travel agency sector.

### 1.4.2 Limitations of the Study

This study has several limitations due to practical, technical, and resource-related constraints:

- The deployment of the chatbot will be limited to free-tier hosting using the Google Cloud Platform, which supports Python-based web applications. This approach enables a faster and more cost-effective development process by avoiding the need to build infrastructure from scratch.

- The Large Language Model (LLM) used for generating responses will be GPT-4.0-mini via the OpenAI API. This model is chosen for its balance between cost-efficiency and performance in delivering high-quality responses in the chatbot.

- The embedding model used for document retrieval in the RAG pipeline is OpenAI’s text-embedding-3-large, selected for its strong semantic understanding, which helps improve the relevance of matched visa-related documents and user inquiries.

- To handle query simplification before retrieval, the study will fine-tune a pretrained T5-small model, which is lightweight and well-suited for environments with limited CPU and memory resources. This model was chosen because it offers acceptable performance while remaining resource-efficient for the computing limitations available to the researchers.

These limitations constrain the scalability and computational intensity of the solution but demonstrate the feasibility of implementing a specialized, intelligent chatbot using accessible and cost-effective technologies.

## 1.5 Significance of the Study
This study aims to introduce an RAG-based chatbot integrated into Messenger to handle inquiries about Japan visa requirements. The significance of this research lies in its potential to enhance operational efficiency, customer satisfaction, and the body of knowledge in RAG-LLM.

1. For Employees: The chatbot will reduce the workload on customer service representatives by automating repetitive inquiries, allowing them to focus on more complex tasks. This saves time and improves resource allocation.

2. For Customers: Customers will experience faster and more accurate responses, improving their overall satisfaction. The chatbot's 24/7 availability ensures that crucial information about visa requirements is accessible at any time, enhancing customers retention.

3. For Developers and Researchers: This study contributes valuable insights into the application of Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) in real-world scenarios. The findings can inform the significance of developing a chatbot in the customer service sector.

## 1.6 Research Questions
To build an effective prototype chatbot for answering Japan visa-related inquiries, this study seeks to address the following research questions:

1. How does the query simplification model affect the performance of retrieving relevant documents?

2. How does fine-tuning the generator model influence the accuracy and relevance of the chatbot’s responses?

3. How do travel agency personnel perceive the usefulness and effectiveness of the chatbot in handling Japan visa-related inquiries, based on their experience using the system?