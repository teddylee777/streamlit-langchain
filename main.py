import streamlit as st
import settings

st.title("📝 나만의 ChatGPT 만들기")

config = settings.load_config()
if "api_key" in config:
    st.session_state.api_key = config["api_key"]

main_text = st.empty()

if "api_key" in st.session_state:
    main_text.markdown(
        f"""저장된 OPENAI API KEY
                
                {st.session_state.api_key}
                
    """
    )
else:
    main_text.markdown(
        f"""저장된 `OPENAI API KEY` 가 없습니다.

🔗 [OPENAI API Key](https://platform.openai.com/account/api-keys)에서 API Key를 발급받을 수 있습니다."""
    )


api_key = st.text_input("🔑 새로운 OPENAI API Key", type="password")

save_btn = st.button("설정 저장", key="save_btn")

if save_btn:
    main_text.markdown(
        f"""저장된 OPENAI API KEY
                
                {api_key}
                
    """
    )
    settings.save_config({"api_key": api_key})
    st.session_state.api_key = api_key
    st.write("설정이 저장되었습니다.")
