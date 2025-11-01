if user_input := st.chat_input("메세지를 입력해 주세요"):
    add_history("user", user_input)
    st.chat_message("user").write(f"{user_input}") 
    with st.chat_message("assistant"):    
        
        llm = RemoteRunnable("https://dioramic-corrin-undetractively.ngrok-free.dev/llm")
        chat_container = st.empty()
        
        if st.session_state.processComplete == True:
            prompt1 = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            retriever = st.session_state['retriever']
            
            # StrOutputParser() 제거!
            rag_chain = (
               {
                   "context": retriever | format_docs,
                   "question": RunnablePassthrough(),
               }
               | prompt1
               | llm
            )
          
            answer = rag_chain.stream(user_input)  
            chunks = []
            for chunk in answer:
               # chunk에서 content 추출
               text = chunk.content if hasattr(chunk, 'content') else str(chunk)
               chunks.append(text)
               chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
            
        else:
            prompt2 = ChatPromptTemplate.from_template(
                "다음의 질문에 간결하게 답변해 주세요:\n{input}"
            )

            # StrOutputParser() 제거!
            chain = prompt2 | llm

            answer = chain.stream(user_input)
            chunks = []
            for chunk in answer:
               # chunk에서 content 추출
               text = chunk.content if hasattr(chunk, 'content') else str(chunk)
               chunks.append(text)
               chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))
