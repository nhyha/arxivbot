import os
import time
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import asyncio
import pdfkit
config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')
import requests
from bs4 import BeautifulSoup
from slack_sdk import WebClient
import discord
import tiktoken
import git

from settings import *
from gpt3 import get_openai_summarization

import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)  #INFO , DEBUG

base_dir = os.path.dirname(os.path.abspath(__file__))  # os.getcwd()
old_paper_set_path = os.path.join(base_dir, "old_paper_set_{}.pickle")
paper_abstracts_path = os.path.join(base_dir, "paper_abstracts.pickle")
#paper_summarizations_path = os.path.join(base_dir, "paper_summarizations.pickle")
paper_instructions_path = os.path.join(base_dir, "paper_instructions.pickle")
paper_full_contents_path = os.path.join(base_dir, "paper_full_contents.pickle")
encoding = tiktoken.encoding_for_model(MODEL)
# encoding = tiktoken.get_encoding(MODEL)

summaries_dir = os.path.join(base_dir, "summaries")
today_summaries_dir = os.path.join(summaries_dir, time.strftime("%Y-%m-%d"))
os.makedirs(today_summaries_dir, exist_ok=True)

# Load environment variables
load_dotenv()
# DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

def get_old_paper_set(workspace):
    if os.path.exists(old_paper_set_path.format(workspace)):
        with open(old_paper_set_path.format(workspace), "rb") as fp:
            old_paper_set = pickle.load(fp)
    else:
        old_paper_set = set()
    return old_paper_set


def get_paper_abstracts():
    if os.path.exists(paper_abstracts_path):
        with open(paper_abstracts_path, "rb") as fp:
            paper_abstracts = pickle.load(fp)
    else:
        paper_abstracts = defaultdict(str)
    return paper_abstracts


def get_paper_instructions():
    if os.path.exists(paper_instructions_path):
        with open(paper_instructions_path, "rb") as fp:
            paper_instructions = pickle.load(fp)
    else:
        paper_instructions = defaultdict(str)
    return paper_instructions


def get_paper_full_contents():
    if os.path.exists(paper_full_contents_path):
        with open(paper_full_contents_path, "rb") as fp:
            paper_full_contents = pickle.load(fp)
    else:
        paper_full_contents = defaultdict(str)
    return paper_full_contents


def get_paper_set_of(field):
    paper_set = []

    # list_url = "https://arxiv.org/list/{}/recent".format(field)
    list_url = "https://arxiv.org/list/{}/pastweek?skip=0&show={}".format(
        field, MAX_NB_CRAWL
    )
    for trial in range(MAX_NB_GPT3_ATTEMPT):
        try:
            list_page = requests.get(list_url)
            if list_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            print(e)
            time.sleep(trial * 30 + 15)
    list_soup = BeautifulSoup(list_page.text, "html.parser")

    # <dd> 태그 추출
    dt_tags = list_soup.find_all("dt")
    dd_tags = list_soup.find_all("dd")

    papers_url = []
    papers_title = []
    papers_comment = []
    for dt_tag, dd_tag in zip(dt_tags, dd_tags):
        # <dd> 태그 내에서 <span class="list-identifier"> 태그 추출
        # identifier_tag = dt_tag.find("span", class_="list-identifier")
        paper_url = dt_tag.find("a", {"title": "Abstract"})["href"]
        paper_url = "https://arxiv.org" + paper_url
        papers_url.append(paper_url)

        # <dd> 태그 내에서 <div class="list-title"> 태그 추출
        title_tag = dd_tag.find("div", class_="list-title")
        paper_title = title_tag.text.strip().strip("Title:").strip()
        papers_title.append(paper_title)

        # <dd> 태그 내에서 <div class="list-comments"> 태그 추출
        comment_tag = dd_tag.find("div", class_="list-comments")
        if comment_tag:
            papers_comment.append(comment_tag.text.strip())
        else:
            papers_comment.append("")

    papers = list(zip(papers_url, papers_title, papers_comment))

    return papers


def get_paper_info(paper_url, paper_title):
    return "{} ({})".format(paper_title, paper_url)


def get_paper_abstract(paper_url):
    for trial in range(MAX_NB_GPT3_ATTEMPT):
        try:
            paper_page = requests.get(paper_url)
            if paper_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            print(e)
            time.sleep(trial * 30 + 15)
    paper_soup = BeautifulSoup(paper_page.text, "html.parser")

    paper_abstract = (
        paper_soup.find_all("blockquote", class_="abstract")[0]
        .text.strip()
        .replace("Abstract:  ", "")
        .replace("\n", " ")
    )
    return paper_abstract


def get_html_experimental_link(paper_url):
    response = requests.get(paper_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # 'HTML (experimental)' 링크 찾기
    html_link = soup.find("a", string="HTML (experimental)")
    if html_link:
        return html_link["href"]  # 링크 추출
    else:
        return "Link not found"


def get_paper_full_content(paper_url):
    for trial in range(MAX_NB_GPT3_ATTEMPT):
        try:
            paper_page = requests.get(paper_url)
            if paper_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            print(e)
            time.sleep(trial * 30 + 15)
    paper_soup = BeautifulSoup(paper_page.text, "html.parser")

    sections = paper_soup.find_all("section")
    section_dict = {}

    for section in sections:
        section_id = section.get("id")
        if section_id:
            # <h2> 태그 내에서 제목 찾기
            title_tag = section.find("h2")
            if title_tag:
                # <span> 태그 내용 제거
                if title_tag.find("span"):
                    title_tag.span.decompose()
                section_title = title_tag.text.strip()
            else:
                section_title = "No title found"

            # 섹션의 전체 텍스트 내용을 추출 (제목 제외)
            section_content = "\n".join(
                [para.text.strip() for para in section.find_all("p")]
            )

            # 사전에 섹션 ID, 제목, 내용 저장
            section_dict[section_id] = {
                "title": section_title,
                "content": section_content,
            }

    # HTML 내용 저장
    html_content = str(paper_soup)
    return section_dict, html_content


def has_new_papers(new_papers, old_paper_set):
    for paper_url, paper_title, _ in new_papers:
        paper_info = get_paper_info(paper_url, paper_title)
        if paper_info not in old_paper_set:
            return True
    return False


def truncate_text(text):
    return encoding.decode(
        encoding.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )[:MAX_INPUT_TOKENS_FOR_SUMMARIZATION]
    )


def prepare_content(paper_info, paper_comment, paper_instructions):
    # 함수의 시작 부분에 다음 코드 추가
    message_content = f"**{paper_info}**\n"
    file_content = f"### {paper_info}\n"
    if paper_comment:
        paper_comment = paper_comment.strip()
        message_content += f"\n{paper_comment}"
        file_content += f"{paper_comment}\n\n"
    paper_instruction_raw = paper_instructions.get(paper_info, "")
    
    if not paper_instruction_raw:
        print(f"Warning: No instruction found for {paper_info}")
        return None, None
    
    try:
        paper_instruction = json.loads(paper_instruction_raw)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError for {paper_info}: {e}")
        print(f"Invalid JSON: {paper_instruction_raw}")
        return None, None
    
    if isinstance(paper_instruction, list):
        paper_instruction = paper_instruction[0]
    
    for key, value in paper_instruction.items():
        message_content += f"\n\n- **{key}**: {value}"
        file_content += f"- **{key}**: {value}\n\n"
    print(f"Prepared content for {paper_info}: message_content={message_content[:50]}..., file_content={file_content[:50]}...")

    return message_content, file_content
    # paper_instruction = json.loads(paper_instructions[paper_info])
    # if type(paper_instruction) is list:
    #     paper_instruction = paper_instruction[0]
    # for key, value in paper_instruction.items():
    #     message_content += f"\n\n- **{key}**: {value}"
    #     file_content += f"- **{key}**: {value}\n\n"

    # return message_content, file_content
    
def save_paper_as_pdf(paper_url, paper_title, html_content):
    # PDF 파일 이름 생성
    pdf_filename = f"{paper_title.replace(' ', '_')[:50]}.pdf"
    pdf_path = os.path.join(base_dir, "pdf_papers", pdf_filename)
    
    # PDF 저장 디렉토리 생성
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    options = {
        'quiet': '',
        'no-outline': None,
        'encoding': "UTF-8",
        'enable-local-file-access': None,
        'disable-smart-shrinking': '',
        #'ignore-ssl-errors': '',
        #'ssl-protocol': 'any',  # SSL 프로토콜 설정
        'no-stop-slow-scripts': None,  # 느린 스크립트 중지 방지
        'disable-external-links': None,  # 외부 링크 비활성화
        'custom-header': [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')],
    }
    options['load-error-handling'] = 'ignore'
    options['javascript-delay'] = '10000'  # 10초 대기
    options['no-stop-slow-scripts'] = None

    # HTML을 PDF로 변환
    try:
        pdfkit.from_html(html_content, pdf_path, options=options) #, configuration=config) #, verbose=True)   #from_url paper_url or from_html html_content
        print(f"Saved PDF: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"Error saving PDF for {paper_title}: {e}")
        return None

def crawl_arxiv(field):
    # collect new arXiv papers
    print("- Collecting new arXiv papers...")
    paper_set = get_paper_set_of(field)

    # abstract crawling
    print("- Crawling the abstracts of papers ...")
    paper_abstracts = get_paper_abstracts()
    paper_full_contents = get_paper_full_contents()
    for paper_url, paper_title, _ in tqdm(paper_set):
        # remove duplicates
        paper_info = get_paper_info(paper_url, paper_title)
        if paper_info in paper_abstracts:
            continue

        paper_abstract = get_paper_abstract(paper_url)
        paper_abstracts[paper_info] = paper_abstract

        with open(paper_abstracts_path, "wb") as fp:
            pickle.dump(paper_abstracts, fp)

        # remove duplicates
        if paper_info in paper_full_contents:
            continue

        html_experimental_link = get_html_experimental_link(paper_url)
        if html_experimental_link != "Link not found":
            paper_full_content, html_content = get_paper_full_content(html_experimental_link)
            paper_full_contents[paper_info] = paper_full_content

            # PDF로 저장
            pdf_path = save_paper_as_pdf(paper_url, paper_title, html_content)
            if pdf_path:
                paper_full_contents[paper_info]['pdf_path'] = pdf_path

            with open(paper_full_contents_path, "wb") as fp:
                pickle.dump(paper_full_contents, fp)

    return paper_set, paper_abstracts, paper_full_contents


def summarize_arxiv(paper_set, paper_abstracts, paper_full_contents):
    print(f"- Generating instructions for the papers by {MODEL}...")
    paper_instructions = get_paper_instructions()

    all_papers = list(paper_set)
    for i in tqdm(range(0, len(all_papers), NB_THREADS)):
        subset_papers = all_papers[i : i + NB_THREADS]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            for paper_url, paper_title, _ in subset_papers:
                # remove duplicates
                paper_info = get_paper_info(paper_url, paper_title)
                if paper_info not in paper_instructions or paper_instructions[paper_info] == "":
                
                    paper_abstract = paper_abstracts[paper_info]
                    paper_full_content = paper_full_contents[paper_info]

                    instruction_input = f"Abstract: {paper_abstract}\n\n"
                    if isinstance(paper_full_content, dict):
                        for _, section in paper_full_content.items():
                            if section["title"] != "No title found" and section["content"]:
                                instruction_input += f"Section: {section['title']}\n{section['content']}\n\n"

                    instruction_input = truncate_text(instruction_input)
                    futures[executor.submit(get_openai_summarization, instruction_input)] = paper_info


            for future in concurrent.futures.as_completed(futures):
                paper_info = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, str):
                        paper_instructions[paper_info] = result
                    else:
                        print(f"Unexpected result type for {paper_info}: {type(result)}")
                except Exception as e:
                    print(f"Error processing paper {paper_info}: {e}")

        with open(paper_instructions_path, "wb") as fp:
            pickle.dump(paper_instructions, fp)

    return paper_instructions
        #         futures[
        #             executor.submit(get_openai_summarization, instruction_input)
        #         ] = paper_info

        #     for f in concurrent.futures.as_completed(futures):
        #         paper_instructions[futures[f]] = f.result()

        # with open(paper_instructions_path, "wb") as fp:
        #     pickle.dump(paper_instructions, fp)

async def send_discord_messages(client, workspace, threads, old_paper_set, workspace_name):
    await client.wait_until_ready()
    guild = client.get_guild(workspace["guild_id"])
    if guild:
        print("Available channels in the guild:")
        for channel in guild.text_channels:
            print(f"- {channel.name}")

        channel = discord.utils.get(
            guild.text_channels, name=workspace["allowed_channel"]
        )
        if channel:
            for thread in threads:
                print(thread["thread_title"].strip())
                main_message = await channel.send(thread["thread_title"])
                thread_obj = await main_message.create_thread(
                    name=thread["thread_title"], auto_archive_duration=1440
                )
                for content in tqdm(thread["thread_contents"]):
                    message_content = content.get("message_content")
                    if not message_content:
                        print(f"Skipping empty message content for {content['paper_info']}")
                        continue

                    message_content += "\n\n"
                    # 메시지 내용을 2000자 이하로 나눔
                    while len(message_content) > 2000:
                        split_index = message_content[:2000].rfind("\n")
                        if split_index == -1 or split_index == 0:
                            split_index = 2000
                        await thread_obj.send(message_content[:split_index].strip())
                        message_content = message_content[split_index:].strip()

                    if message_content:
                        await thread_obj.send(message_content.strip())

                    # pickling after messaging
                    old_paper_set.add(content["paper_info"])
                    with open(old_paper_set_path.format(workspace_name), "wb") as fp:
                        pickle.dump(old_paper_set, fp)
        else:
            raise Exception(f"Channel {workspace['allowed_channel']} not found.")
    else:
        raise Exception(f"Guild {workspace['guild_id']} not found.")

    await client.close()


# async def send_discord_messages(
#     client, workspace, threads, old_paper_set, workspace_name
# ):
#     await client.wait_until_ready()
#     guild = client.get_guild(workspace["guild_id"])
#     if guild:
#         channel = discord.utils.get(
#             guild.text_channels, name=workspace["allowed_channel"]
#         )
#         if channel:
#             for thread in threads:
#                 print(thread["thread_title"].strip())
#                 main_message = await channel.send(thread["thread_title"])
#                 thread_obj = await main_message.create_thread(
#                     name=thread["thread_title"], auto_archive_duration=1440
#                 )
#                 for content in tqdm(thread["thread_contents"]):
#                     await thread_obj.send(content["message_content"] + "\n\n")
#                     # pickling after messaging
#                     old_paper_set.add(content["paper_info"])
#                     with open(old_paper_set_path.format(workspace_name), "wb") as fp:
#                         pickle.dump(old_paper_set, fp)
#         else:
#             raise Exception(f"Channel {workspace['allowed_channel']} not found.")
#     else:
#         raise Exception(f"Guild {workspace['guild_id']} not found.")

#     await client.close()


def main():
    # 전체 흐름은 다음과 같습니다.
    # 1. arXiv의 새 논문을 수집합니다.
    # 2. 수집한 논문의 초록을 크롤링합니다.
    # 3. 수집한 논문의 초록을 GPT-4o로 요약합니다.
    # 4. 요약된 내용을 Slack 또는 Discord에 전송합니다.
    fields = set()
    for workspace in WORKSPACES:
        for field in workspace["fields"]:
            fields.add(field)

    new_papers = defaultdict(list)
    for field in fields:
        print("Processing {} field...".format(field))
        paper_set, paper_abstracts, paper_full_contents = crawl_arxiv(field)
        summarize_arxiv(paper_set, paper_abstracts, paper_full_contents)

        new_papers[field] = paper_set

    paper_instructions = get_paper_instructions()

    for workspace in WORKSPACES:
        workspace_name = f"{workspace['workspace']}-{workspace['allowed_channel']}"

        # prepare messages
        threads = []
        old_paper_set = get_old_paper_set(workspace_name)
        for field in tqdm(workspace["fields"]):
            if not has_new_papers(new_papers[field], old_paper_set):
                continue

            thread = {
                "thread_title": f"New uploads on arXiv({field})\n",
                "thread_contents": [],
            }

            # make a thread by replying to the parent message
            for paper_url, paper_title, paper_comment in new_papers[field]:
                paper_info = get_paper_info(paper_url, paper_title)

                # remove duplicates
                if paper_info in old_paper_set:
                    continue

                message_content, file_content = prepare_content(
                    paper_info,
                    paper_comment,
                    paper_instructions,
                )

                if message_content and file_content:
                    print(f"Appending content for {paper_info}")
                    thread["thread_contents"].append(
                        {
                            "paper_info": paper_info,
                            "message_content": message_content,
                            "file_content": file_content,
                        }
                    )
                else:
                    print(f"Skipping content for {paper_info} due to empty message or file content")

            threads.append(thread)

        # send messages
        print("Sending messages...")
        # if workspace["service_type"] == "slack":
        #     print("Connecting", workspace["workspace"], "...")
        #     sc = WebClient(workspace["slack_token"])

        #     nb_messages = 0
        #     for thread in threads:
        #         sc.chat_postMessage(
        #             channel=workspace["allowed_channel"],
        #             text=thread["thread_title"],
        #         )

        #         # get the timestamp of the parent message
        #         result = sc.conversations_history(
        #             channel=workspace["allowed_channel_id"]
        #         )
        #         conversation_history = result[
        #             "messages"
        #         ]  # [0] is the most recent message
        #         message_ts = conversation_history[0]["ts"]

        #         for content in tqdm(thread["thread_contents"]):
        #             sc.chat_postMessage(
        #                 channel=workspace["allowed_channel"],
        #                 text=content["message_content"],
        #                 thread_ts=message_ts,
        #             )

        #             # pickling after messaging
        #             old_paper_set.add(content["paper_info"])
        #             with open(old_paper_set_path.format(workspace_name), "wb") as fp:
        #                 pickle.dump(old_paper_set, fp)

        #             nb_messages += 1
        #             if nb_messages >= MAX_NB_SHOW:
        #                 nb_messages = 0
        #                 time.sleep(TIME_PAUSE_SEC)

        # 왜 discord.py 라이브러리는 async 만 된다는거야 대체
        if workspace["service_type"] == "discord":
            print("Connecting", workspace["workspace"], "...")
            intents = discord.Intents.default()
            intents.messages = True

            client = discord.Client(intents=intents)

            loop = asyncio.get_event_loop()
            loop.create_task(client.start(workspace["discord_token"]))
            loop.run_until_complete(
                send_discord_messages(
                    client, workspace, threads, old_paper_set, workspace_name
                )
            )



        for field in fields:
            today_summaries_field_path = os.path.join(today_summaries_dir, field + ".md")
            logging.info(f"Attempting to write summary for field: {field}")
            logging.info(f"File path: {today_summaries_field_path}")
            
            try:
                with open(today_summaries_field_path, "w", encoding="utf-8") as fp:
                    thread_count = 0
                    content_count = 0
                    for thread in threads:
                        fp.write(thread["thread_title"] + "\n")
                        thread_count += 1
                        for content in thread["thread_contents"]:
                            fp.write(content.get("file_content", "") + "\n\n")
                            content_count += 1
                
                logging.info(f"Successfully wrote summary for {field}. Threads: {thread_count}, Contents: {content_count}")
            except Exception as e:
                logging.error(f"Error writing summary for {field}: {str(e)}")

            # 파일이 실제로 생성되었는지 확인
            if os.path.exists(today_summaries_field_path):
                logging.info(f"Confirmed: File {field}.md was created.")
            else:
                logging.warning(f"Warning: File {field}.md was not created.")

    repo = git.Repo(base_dir)
    repo.git.add(os.path.join(base_dir, "summaries"))
    repo.git.commit("-m", f"\"Update summaries: {time.strftime('%Y-%m-%d')}\"")
    repo.git.push(force=True)


if __name__ == "__main__":
    main()
